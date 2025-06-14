#!/usr/bin/env python3
"""
Quick test script for the Photo Processor API
"""

import httpx
import asyncio
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

async def test_api():
    async with httpx.AsyncClient() as client:
        print("Testing Photo Processor API...\n")
        
        # Test root endpoint
        print("1. Testing root endpoint...")
        response = await client.get(f"{API_BASE}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        response = await client.get(f"{API_BASE}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test dashboard stats
        print("\n3. Testing dashboard stats...")
        response = await client.get(f"{API_BASE}/api/stats/dashboard")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        # Test processing status
        print("\n4. Testing processing status...")
        response = await client.get(f"{API_BASE}/api/processing/status")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        # Test recipe presets
        print("\n5. Testing recipe presets...")
        response = await client.get(f"{API_BASE}/api/recipes/presets/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        # Test WebSocket connection
        print("\n6. Testing WebSocket connection...")
        try:
            import websockets
            async with websockets.connect(f"ws://localhost:8000/ws") as websocket:
                # Wait for connection message
                message = await websocket.recv()
                print(f"   WebSocket connected!")
                print(f"   Message: {message}")
                await websocket.close()
        except Exception as e:
            print(f"   WebSocket error: {e}")
        
        print("\nAPI tests completed!")

if __name__ == "__main__":
    asyncio.run(test_api())