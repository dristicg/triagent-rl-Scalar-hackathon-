#!/usr/bin/env python3
"""
Quick verification script for TriageNet-RL
Tests core imports and basic functionality without external dependencies.
"""

import sys
import os

def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from models import Action, Observation, Patient, Vitals, BedStatus, Reward, StepResult, EpisodeResult
        print("✅ models.py - All classes imported")
    except Exception as e:
        print(f"❌ models.py import failed: {e}")
        return False
    
    try:
        from tasks import ALL_PATIENTS, TASK_CONFIGS
        print(f"✅ tasks.py - {len(ALL_PATIENTS)} patients, {len(TASK_CONFIGS)} tasks")
    except Exception as e:
        print(f"❌ tasks.py import failed: {e}")
        return False
    
    try:
        from env import MedicalTriageEnv
        print("✅ env.py - MedicalTriageEnv imported")
    except Exception as e:
        print(f"❌ env.py import failed: {e}")
        return False
    
    try:
        from baseline import RuleBasedAgent
        print("✅ baseline.py - RuleBasedAgent imported")
    except Exception as e:
        print(f"❌ baseline.py import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic environment functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from models import Action, Vitals, Patient
        from env import MedicalTriageEnv
        
        # Create environment
        env = MedicalTriageEnv(task_id=1, seed=42)
        print("✅ Environment created")
        
        # Reset environment
        obs = env.reset()
        print(f"✅ Environment reset - Task: {obs.task_id}, Pending: {len(obs.pending_patients)}")
        
        # Create action
        if obs.pending_patients:
            action = Action(
                action_type="assign_esi",
                patient_id=obs.pending_patients[0],
                esi_level=2,
                routing_zone="acute_care"
            )
            print("✅ Action created")
            
            # Step environment
            result = env.step(action)
            print(f"✅ Step executed - Reward: {result.reward.step_reward:+.3f}")
            
            # Grade episode
            if result.done:
                grade = env.grade()
                print(f"✅ Episode graded - Score: {grade.final_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Verify all required files exist."""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "models.py",
        "tasks.py", 
        "env.py",
        "inference.py",
        "baseline.py",
        "openenv.yaml",
        "openenv_validate.py",
        "app.py",
        "requirements.txt",
        "Dockerfile",
        "README.md"
    ]
    
    rl_files = [
        "rl/__init__.py",
        "rl/feature_extractor.py",
        "rl/action_mapper.py", 
        "rl/environment_wrapper.py",
        "rl/train_ppo.py",
        "rl/evaluate.py",
        "rl/requirements_rl.txt",
        "rl/README_RL.md"
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    for file in rl_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    return len(missing_files) == 0

def main():
    """Run all verification tests."""
    print("🏥 TriageNet-RL Verification")
    print("=" * 50)
    
    # Run tests
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    structure_ok = test_file_structure()
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    if imports_ok:
        print("✅ All imports successful")
    else:
        print("❌ Some imports failed")
    
    if functionality_ok:
        print("✅ Basic functionality works")
    else:
        print("❌ Basic functionality failed")
    
    if structure_ok:
        print("✅ All required files present")
    else:
        print("❌ Some files missing")
    
    all_passed = imports_ok and functionality_ok and structure_ok
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("TriageNet-RL is ready for use!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run baseline: python baseline.py --all")
        print("3. Launch demo: python app.py")
        print("4. Run validation: python openenv_validate.py")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
