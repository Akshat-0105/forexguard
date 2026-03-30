import asyncio
import pandas as pd
import aiohttp
import os

API_URL = "http://localhost:8000/predict"


class StreamingSimulator:
    def __init__(self, file_path, rate_per_sec=20, workers=5):
        self.file_path = file_path
        self.rate = rate_per_sec
        self.workers = workers
        self.queue = asyncio.Queue(maxsize=1000)

    def _clean_event(self, row):
        def clean_str(val, default="unknown"):
            if pd.isna(val) or val is None or str(val).strip().lower() == "nan":
                return default
            return str(val)

        def clean_float(val):
            if pd.isna(val) or val is None:
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        return {
            "user_id": clean_str(row.get("user_id")),
            "timestamp": clean_str(row.get("timestamp")),
            "event_type": clean_str(row.get("event_type")),
            "ip_address": clean_str(row.get("ip_address"), default="0.0.0.0"),
            "device_id": clean_str(row.get("device_id")),
            "amount": clean_float(row.get("amount")),
            "trade_volume": clean_float(row.get("trade_volume")),
            "instrument": clean_str(row.get("instrument")),
            "session_id": clean_str(row.get("session_id")),
        }

    async def producer(self):
        if not os.path.exists(self.file_path):
            print(f"ERROR: File not found at {self.file_path}")
            return

        df = pd.read_csv(self.file_path).sample(100, random_state=42)
        interval = 1.0 / self.rate

        print(f"Streaming {len(df)} events at {self.rate} req/s")

        for i, (_, row) in enumerate(df.iterrows()):
            event = self._clean_event(row.to_dict())
            await self.queue.put(event)
            await asyncio.sleep(interval)
            if i % 10 == 0:
                print(f"[Producer] {i} events queued")

        print("Producer finished")

    async def worker(self, session, worker_id):
        print(f"[Worker {worker_id}] started")

        while True:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=5)
            except asyncio.TimeoutError:
                continue

            try:
                async with session.post(API_URL, json=event) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        print(f"[Worker {worker_id}] API {resp.status}: {error}")
                        continue

                    result = await resp.json()

                    score = result.get("anomaly_score", 0.0)
                    pred = result.get("prediction", 0)

                    print(f"[Worker {worker_id}] Score={score:.3f}, Pred={pred}")

            except Exception as e:
                print(f"[Worker {worker_id}] Error:", e)

            finally:
                self.queue.task_done()

    async def run(self):
        async with aiohttp.ClientSession() as session:
            workers = [
                asyncio.create_task(self.worker(session, i))
                for i in range(self.workers)
            ]

            producer = asyncio.create_task(self.producer())

            await producer
            await self.queue.join()

            for w in workers:
                w.cancel()

            print("Streaming complete")


if __name__ == "__main__":
    simulator = StreamingSimulator(
        file_path="data/synthetic_events.csv",
        rate_per_sec=20,   # stable
        workers=5
    )

    try:
        asyncio.run(simulator.run())
    except KeyboardInterrupt:
        print("Stopped")