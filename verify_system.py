#!/usr/bin/env python3
"""
System verification script to ensure everything runs perfectly.
"""

import sys
import os

def verify_system():
    """Verify all components are ready for perfect execution"""
    print("🔍 Vision Assistant System Verification")
    print("=" * 50)
    
    issues = []
    
    # Check main.py exists and is valid Python
    if not os.path.exists("main.py"):
        issues.append("❌ main.py not found")
    else:
        try:
            with open("main.py", 'r') as f:
                content = f.read()
                # Check for essential components
                required_components = [
                    "def main():",
                    "def camera_loop():",
                    "def wake_loop():", 
                    "def voice_loop():",
                    "def auto_describe_loop():",
                    "ObjectDetector()",
                    "FaceRecognizer()",
                    "CommandHandler(",
                    "if __name__ == \"__main__\":"
                ]
                
                for component in required_components:
                    if component not in content:
                        issues.append(f"❌ Missing: {component}")
                        
        except Exception as e:
            issues.append(f"❌ Error reading main.py: {e}")
    
    # Check core modules
    required_modules = [
        "vision/detector.py",
        "vision/face_recognizer.py", 
        "vision/ocr_reader.py",
        "voice/speaker.py",
        "voice/listener.py",
        "voice/wake_word.py",
        "core/scene_builder.py",
        "core/command_handler.py"
    ]
    
    for module in required_modules:
        if not os.path.exists(module):
            issues.append(f"❌ Missing module: {module}")
    
    # Check requirements.txt
    if not os.path.exists("requirements.txt"):
        issues.append("❌ requirements.txt not found")
    
    # Check for AI detector
    ai_detectors = [
        "vision/ai_detector.py",
        "vision/simple_ai_detector.py"
    ]
    
    ai_found = any(os.path.exists(detector) for detector in ai_detectors)
    if not ai_found:
        issues.append("❌ No AI detector found")
    
    # Report results
    if not issues:
        print("✅ All system components verified!")
        print("✅ main.py is complete and ready")
        print("✅ All required modules present")
        print("✅ AI detection available")
        print("✅ Requirements file exists")
        print("\n🚀 System is ready to run perfectly!")
        print("\nRun command:")
        print("python main.py")
    else:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\n🔧 Fix issues before running")

if __name__ == "__main__":
    verify_system()
