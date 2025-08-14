#!/usr/bin/env python3
"""
Demo script for testing Oumi WebChat functionality.

This script demonstrates how to launch the WebChat interface with a simple model.
"""

import os
import sys
import time

# Add the src directory to path for development
sys.path.insert(0, 'src')

def main():
    """Run the WebChat demo."""
    print("🚀 Oumi WebChat Demo")
    print("=" * 50)
    
    # Check if webchat dependencies are available
    try:
        import gradio as gr
        print("✅ Gradio available")
    except ImportError:
        print("❌ Gradio not available. Install with:")
        print("   pip install 'oumi[webchat]'")
        return False
        
    try:
        import aiohttp_cors
        print("✅ aiohttp-cors available")
    except ImportError:
        print("❌ aiohttp-cors not available. Install with:")
        print("   pip install 'oumi[webchat]'")  
        return False
    
    # Test basic imports
    try:
        from oumi.webchat import create_webchat_interface, OumiWebServer
        print("✅ WebChat modules imported successfully")
    except Exception as e:
        print(f"❌ Error importing WebChat modules: {e}")
        return False
        
    try:
        from oumi.core.configs import InferenceConfig
        print("✅ InferenceConfig imported successfully")
    except Exception as e:
        print(f"❌ Error importing InferenceConfig: {e}")
        return False
    
    # Create a minimal config for testing
    # This would normally be loaded from a YAML file
    print("\n📝 Creating test configuration...")
    
    # For the demo, we'll create a basic config
    # In practice, you would use: oumi webchat -c path/to/your/config.yaml
    test_config_content = """
# Minimal test configuration for WebChat demo
model:
  model_name: "test-model"
  model_max_length: 4096
  torch_dtype_str: "float16"

generation:
  max_new_tokens: 1024
  temperature: 0.7

engine: NATIVE

style:
  use_emoji: true
  expand_panels: true
"""
    
    # Write test config
    with open("demo_config.yaml", "w") as f:
        f.write(test_config_content)
    print("✅ Created demo_config.yaml")
    
    print("\n🌐 To test WebChat, run:")
    print("   python demo_webchat.py --run")
    print("   # or use the CLI:")
    print("   oumi webchat -c demo_config.yaml")
    print("\n📖 This will start:")
    print("   • Backend server at http://localhost:8000")  
    print("   • Frontend interface at http://localhost:7860")
    print("\n💡 Features to test:")
    print("   • Basic chat interface")
    print("   • Interactive branch tree")
    print("   • Command execution (/help, /clear, etc.)")
    print("   • File attachments")
    print("   • System monitoring")
    
    # If --run is passed, actually launch the demo
    if "--run" in sys.argv:
        print("\n🚀 Launching WebChat demo...")
        launch_demo()
    
    return True

def launch_demo():
    """Launch the actual WebChat demo."""
    try:
        from oumi.core.configs import InferenceConfig
        from oumi.utils.logging import logger
        
        # Load the demo config
        config = InferenceConfig.from_yaml_file("demo_config.yaml")
        config.finalize_and_validate()
        
        print("📊 Configuration loaded:")
        config.print_config(logger)
        
        # Import and launch
        from oumi.webchat.interface import launch_webchat
        from oumi.webchat.server import run_webchat_server
        import threading
        import time
        
        # Start backend in thread
        def run_backend():
            run_webchat_server(config, host="localhost", port=8000)
            
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(3)
        
        # Launch frontend
        print("🌐 Launching frontend...")
        launch_webchat(
            config=config,
            server_url="http://localhost:8000",
            share=False,
            server_name="localhost", 
            server_port=7860
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)