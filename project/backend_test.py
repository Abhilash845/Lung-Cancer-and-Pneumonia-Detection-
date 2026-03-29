import requests
import sys
import os
from datetime import datetime
import io
from PIL import Image
import numpy as np

class LungDiseaseAPITester:
    def __init__(self, base_url="https://lungchoice-scan.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}" if endpoint else self.base_url
        headers = {}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, timeout=60)
                else:
                    headers['Content-Type'] = 'application/json'
                    response = requests.post(url, json=data, headers=headers, timeout=60)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"Response: {response_data}")
                    return True, response_data
                except:
                    print(f"Response (non-JSON): {response.text[:200]}...")
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"Response: {response.text[:500]}...")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def create_test_image(self):
        """Create a simple test image for upload"""
        # Create a simple 224x224 RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes

    def test_health_check(self):
        """Test health check endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "health",
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   - Status: {response.get('status', 'Unknown')}")
            print(f"   - Pneumonia model loaded: {response.get('pneumonia_model_loaded', 'Unknown')}")
            print(f"   - Lung cancer model loaded: {response.get('lung_cancer_model_loaded', 'Unknown')}")
            
            # Check if models are loaded
            if not response.get('pneumonia_model_loaded', False):
                print("   ⚠️  Warning: Pneumonia model not loaded")
            if not response.get('lung_cancer_model_loaded', False):
                print("   ⚠️  Warning: Lung cancer model not loaded")
        
        return success

    def test_root_endpoint(self):
        """Test root API endpoint"""
        success, response = self.run_test(
            "Root Endpoint",
            "GET",
            "",
            200
        )
        return success

    def test_predict_with_image(self):
        """Test prediction endpoint with a test image"""
        print("\n📸 Creating test image...")
        test_image = self.create_test_image()
        
        files = {
            'file': ('test_xray.jpg', test_image, 'image/jpeg')
        }
        
        success, response = self.run_test(
            "Predict with Test Image",
            "POST",
            "predict",
            200,
            files=files
        )
        
        if success and isinstance(response, dict):
            print(f"   - Pneumonia result: {response.get('pneumonia_result', 'Unknown')}")
            print(f"   - Lung cancer result: {response.get('lung_cancer_result', 'Unknown')}")
            print(f"   - Clinical recommendations count: {len(response.get('clinical_recommendations', []))}")
            print(f"   - Lifestyle recommendations count: {len(response.get('lifestyle_recommendations', []))}")
            print(f"   - Confidence info: {response.get('confidence_info', 'Unknown')}")
            
            # Validate response structure
            required_fields = ['pneumonia_result', 'lung_cancer_result', 'clinical_recommendations', 'lifestyle_recommendations', 'confidence_info']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"   ⚠️  Warning: Missing fields in response: {missing_fields}")
                return False
        
        return success

    def test_predict_with_invalid_file(self):
        """Test prediction endpoint with invalid file"""
        # Create a text file instead of image
        text_content = io.BytesIO(b"This is not an image file")
        
        files = {
            'file': ('test.txt', text_content, 'text/plain')
        }
        
        success, response = self.run_test(
            "Predict with Invalid File",
            "POST",
            "predict",
            400,  # Expecting 400 Bad Request
            files=files
        )
        
        return success

    def test_predict_without_file(self):
        """Test prediction endpoint without file"""
        success, response = self.run_test(
            "Predict without File",
            "POST",
            "predict",
            422,  # Expecting 422 Unprocessable Entity
        )
        
        return success

def main():
    print("🏥 Starting Lung Disease Detection API Tests")
    print("=" * 60)
    
    # Setup
    tester = LungDiseaseAPITester()
    
    # Run tests in order
    print("\n📋 Running API Tests...")
    
    # Basic connectivity tests
    tester.test_root_endpoint()
    tester.test_health_check()
    
    # Main functionality tests
    tester.test_predict_with_image()
    
    # Error handling tests
    tester.test_predict_with_invalid_file()
    tester.test_predict_without_file()
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print(f"❌ {tester.tests_run - tester.tests_passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())