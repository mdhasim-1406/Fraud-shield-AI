#!/usr/bin/env python3
"""
Verification script for Fraud Detection SOC Cockpit
Tests all critical functionality after KeyError fix
"""

import requests
import time
import sys

def test_api_connection():
    """Test if FastAPI backend is running"""
    print("🔍 Testing API Connection...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI backend is online")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_recent_transactions():
    """Test if we can fetch recent transactions"""
    print("\n🔍 Testing Recent Transactions Endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/v1/recent-transactions", timeout=5)
        if response.status_code == 200:
            data = response.json()
            count = data.get('total_count', 0)
            print(f"✅ Retrieved {count} transactions from API")
            if count > 0:
                print(f"   Sample transaction ID: {data['transactions'][0].get('transaction_id', 'N/A')}")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error fetching transactions: {e}")
        return False

def test_stats_endpoint():
    """Test if system stats are available"""
    print("\n🔍 Testing System Stats Endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/v1/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ System stats retrieved successfully")
            print(f"   Total transactions processed: {stats.get('total_transactions_processed', 0)}")
            print(f"   Fraud detection rate: {stats.get('fraud_detection_rate', 0):.1%}")
            print(f"   Average processing time: {stats.get('average_processing_time', 0):.3f}s")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error fetching stats: {e}")
        return False

def test_dashboard_availability():
    """Test if Gradio dashboard is accessible"""
    print("\n🔍 Testing Dashboard Availability...")
    try:
        response = requests.get("http://localhost:7860", timeout=5)
        if response.status_code == 200:
            print("✅ Gradio dashboard is accessible at http://localhost:7860")
            return True
        else:
            print(f"❌ Dashboard returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to dashboard: {e}")
        return False

def main():
    """Run all verification tests"""
    print("="*60)
    print("🛡️  Fraud Detection SOC Cockpit Verification")
    print("="*60)
    
    tests = [
        ("API Connection", test_api_connection),
        ("Recent Transactions", test_recent_transactions),
        ("System Stats", test_stats_endpoint),
        ("Dashboard Availability", test_dashboard_availability)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
        time.sleep(0.5)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! System is operational.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
