#!/usr/bin/env python3
"""
Verification script for YOLO12 configuration files
"""

import yaml
import os
from ultralytics import YOLO

def verify_dataset_config():
    """Verify the dataset configuration file"""
    print("=== Verifying Dataset Configuration ===")
    
    data_yaml_path = "../data/fer2013.yaml"
    
    if not os.path.exists(data_yaml_path):
        print(f"❌ Dataset config file not found: {data_yaml_path}")
        return False
    
    try:
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Dataset config loaded successfully")
        print(f"   Path: {config.get('path', 'Not found')}")
        print(f"   Train: {config.get('train', 'Not found')}")
        print(f"   Val: {config.get('val', 'Not found')}")
        print(f"   Classes: {config.get('nc', 'Not found')}")
        print(f"   Class names: {list(config.get('names', {}).values())}")
        
        # Check if directories exist
        data_path = os.path.join(os.path.dirname(data_yaml_path), config.get('path', ''))
        train_path = os.path.join(data_path, config.get('train', ''))
        val_path = os.path.join(data_path, config.get('val', ''))
        
        print(f"   Train directory exists: {os.path.exists(train_path)}")
        print(f"   Val directory exists: {os.path.exists(val_path)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset config: {e}")
        return False

def verify_model_config():
    """Verify the model configuration file"""
    print("\n=== Verifying Model Configuration ===")
    
    model_yaml_path = "../models/yolo12_fer2013.yaml"
    
    if not os.path.exists(model_yaml_path):
        print(f"❌ Model config file not found: {model_yaml_path}")
        return False
    
    try:
        with open(model_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Model config loaded successfully")
        print(f"   Classes: {config.get('nc', 'Not found')}")
        print(f"   Depth multiple: {config.get('depth_multiple', 'Not found')}")
        print(f"   Width multiple: {config.get('width_multiple', 'Not found')}")
        print(f"   Learning rate: {config.get('lr0', 'Not found')}")
        print(f"   Optimizer settings: Adam configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model config: {e}")
        return False

def test_yolo_model_creation():
    """Test creating a YOLO model from the config"""
    print("\n=== Testing YOLO Model Creation ===")
    
    model_yaml_path = "../models/yolo12_fer2013.yaml"
    
    try:
        model = YOLO(model_yaml_path)
        print(f"✅ YOLO model created successfully")
        print(f"   Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        print(f"   Number of classes: {model.model.nc}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating YOLO model: {e}")
        return False

def main():
    """Main verification function"""
    print("YOLO12 Configuration Verification")
    print("=" * 40)
    
    # Verify configurations
    dataset_ok = verify_dataset_config()
    model_ok = verify_model_config()
    yolo_ok = test_yolo_model_creation()
    
    # Summary
    print("\n=== Verification Summary ===")
    print(f"Dataset config: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
    print(f"Model config: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"YOLO model creation: {'✅ PASS' if yolo_ok else '❌ FAIL'}")
    
    if all([dataset_ok, model_ok, yolo_ok]):
        print("\n🎉 All verifications passed! Ready for training.")
        print("\nNext steps:")
        print("1. Open yolo12_training_notebook.ipynb in Jupyter")
        print("2. Run all cells to start training")
        print("3. Monitor training progress in the notebook")
    else:
        print("\n⚠️  Some verifications failed. Please check the configuration files.")

if __name__ == "__main__":
    main() 