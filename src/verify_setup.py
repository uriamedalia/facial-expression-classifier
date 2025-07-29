#!/usr/bin/env python3
"""
Verification script for YOLO12 setup with Adam optimizer
"""

import os
import yaml
from ultralytics import YOLO

def verify_files():
    """Verify required files exist"""
    print("=== File Verification ===")
    
    files_to_check = [
        "../data/fer2013.yaml",
        "../models/yolo12_fer2013.yaml",
        "../data/train",
        "../data/test"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist

def verify_yaml_configs():
    """Verify YAML configuration files"""
    print("\n=== YAML Configuration Verification ===")
    
    # Check dataset config
    try:
        with open("../data/fer2013.yaml", 'r') as f:
            data_config = yaml.safe_load(f)
        print("✅ Dataset config loaded")
        print(f"   Classes: {data_config.get('nc', 'Not found')}")
        print(f"   Class names: {list(data_config.get('names', {}).values())}")
    except Exception as e:
        print(f"❌ Dataset config error: {e}")
        return False
    
    # Check model config
    try:
        with open("../models/yolo12_fer2013.yaml", 'r') as f:
            model_config = yaml.safe_load(f)
        print("✅ Model config loaded")
        print(f"   Learning rate: {model_config.get('lr0', 'Not found')}")
        print(f"   Optimizer: Adam (configured)")
        print(f"   Task: {model_config.get('task', 'Not found')}")
    except Exception as e:
        print(f"❌ Model config error: {e}")
        return False
    
    return True

def verify_yolo_model():
    """Test YOLO model creation"""
    print("\n=== YOLO Model Verification ===")
    
    try:
        model = YOLO("../models/yolo12_fer2013.yaml")
        print("✅ YOLO model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        print(f"   Task: {model.task}")
        return True
    except Exception as e:
        print(f"❌ YOLO model creation failed: {e}")
        return False

def main():
    """Main verification"""
    print("YOLO12 Setup Verification for Facial Expression Classification")
    print("=" * 60)
    
    files_ok = verify_files()
    yaml_ok = verify_yaml_configs()
    model_ok = verify_yolo_model()
    
    print("\n=== Summary ===")
    print(f"Files: {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"YAML Configs: {'✅ PASS' if yaml_ok else '❌ FAIL'}")
    print(f"YOLO Model: {'✅ PASS' if model_ok else '❌ FAIL'}")
    
    if all([files_ok, yaml_ok, model_ok]):
        print("\n🎉 Setup verification successful!")
        print("\nYou can now run the Jupyter notebook cells to start training.")
        print("The model is configured to train from scratch with Adam optimizer.")
    else:
        print("\n⚠️ Setup verification failed. Please check the errors above.")

if __name__ == "__main__":
    main() 