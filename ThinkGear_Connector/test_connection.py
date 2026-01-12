"""
Test connection to MindWave headset via ThinkGear Connector
Run this before the main experiment to verify everything is working.
"""

import socket
import json
import time

def test_mindwave_connection(host='127.0.0.1', port=13854, duration=10):
    """
    Test connection to MindWave headset
    
    Args:
        host: ThinkGear Connector host (default: localhost)
        port: ThinkGear Connector port (default: 13854)
        duration: How long to test in seconds
    """
    
    print("=" * 50)
    print("MINDWAVE CONNECTION TEST")
    print("=" * 50)
    
    # Step 1: Check if ThinkGear Connector is running
    print("\n[1] Connecting to ThinkGear Connector...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    
    try:
        sock.connect((host, port))
        print("    ✓ Connected to ThinkGear Connector!")
    except ConnectionRefusedError:
        print("    ✗ FAILED: ThinkGear Connector is not running!")
        print("\n    → Start 'ThinkGear Connector.exe' first")
        print("    → Make sure the headset is paired and connected")
        return False
    except socket.timeout:
        print("    ✗ FAILED: Connection timed out!")
        return False
    
    # Step 2: Send configuration
    print("\n[2] Configuring connection...")
    try:
        config = {"enableRawOutput": True, "format": "Json"}
        sock.send(json.dumps(config).encode('utf-8'))
        print("    ✓ Configuration sent!")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        sock.close()
        return False
    
    # Step 3: Test receiving data
    print(f"\n[3] Receiving data for {duration} seconds...")
    print("    (Put on the headset and wait)\n")
    
    sock.setblocking(False)
    
    buffer = ""
    start_time = time.time()
    
    # Counters
    raw_count = 0
    esense_count = 0
    power_count = 0
    blink_count = 0
    poor_signal_count = 0
    last_attention = 0
    last_meditation = 0
    last_signal_quality = 200  # 200 = no contact
    
    try:
        while time.time() - start_time < duration:
            try:
                data = sock.recv(4096).decode('utf-8')
                if data:
                    buffer += data
                    
                    while '\r' in buffer:
                        packet, buffer = buffer.split('\r', 1)
                        try:
                            reading = json.loads(packet)
                            
                            # Count data types
                            if 'rawEeg' in reading:
                                raw_count += 1
                            
                            if 'eSense' in reading:
                                esense_count += 1
                                last_attention = reading['eSense']['attention']
                                last_meditation = reading['eSense']['meditation']
                                print(f"    eSense - Attention: {last_attention:3d} | Meditation: {last_meditation:3d}")
                            
                            if 'eegPower' in reading:
                                power_count += 1
                            
                            if 'blinkStrength' in reading:
                                blink_count += 1
                                print(f"    *** BLINK DETECTED: {reading['blinkStrength']} ***")
                            
                            if 'poorSignalLevel' in reading:
                                last_signal_quality = reading['poorSignalLevel']
                                if last_signal_quality > 0:
                                    poor_signal_count += 1
                                    
                        except json.JSONDecodeError:
                            pass
                            
            except BlockingIOError:
                pass
            except Exception as e:
                print(f"    Error: {e}")
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n    Stopped by user.")
    
    sock.close()
    
    # Step 4: Report results
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    print(f"\nData received in {duration} seconds:")
    print(f"  Raw EEG samples:  {raw_count:6d}  (expected: ~{512*duration})")
    print(f"  eSense readings:  {esense_count:6d}  (expected: ~{duration})")
    print(f"  EEG Power bands:  {power_count:6d}  (expected: ~{duration})")
    print(f"  Blinks detected:  {blink_count:6d}")
    print(f"  Poor signal events: {poor_signal_count}")
    
    print(f"\nLast readings:")
    print(f"  Attention:    {last_attention}")
    print(f"  Meditation:   {last_meditation}")
    print(f"  Signal Quality: {last_signal_quality} (0=good, 200=no contact)")
    
    # Determine success
    print("\n" + "-" * 50)
    
    success = True
    
    if raw_count < 100:
        print("⚠ WARNING: Very few raw samples received!")
        print("  → Check headset contact with forehead")
        success = False
    else:
        print("✓ Raw EEG data: OK")
    
    if esense_count < 2:
        print("⚠ WARNING: No eSense values received!")
        print("  → Headset may not be properly worn")
        success = False
    else:
        print("✓ eSense values: OK")
    
    if last_signal_quality > 50:
        print("⚠ WARNING: Poor signal quality!")
        print("  → Adjust headset position")
        print("  → Make sure ear clip is attached")
        print("  → Clean sensor contacts")
        success = False
    else:
        print("✓ Signal quality: OK")
    
    if last_attention == 0 and last_meditation == 0:
        print("⚠ WARNING: Attention/Meditation stuck at 0")
        print("  → Wait a few more seconds for calibration")
        success = False
    else:
        print("✓ Attention/Meditation: OK")
    
    print("-" * 50)
    
    if success:
        print("\n✓ ALL TESTS PASSED! Ready for experiment.")
    else:
        print("\n✗ SOME ISSUES DETECTED. Fix them before running experiment.")
    
    return success


if __name__ == "__main__":
    print("\nMake sure:")
    print("  1. ThinkGear Connector is running")
    print("  2. Headset is turned on and paired")
    print("  3. Headset is being worn properly")
    print()
    
    input("Press ENTER to start test...")
    print()
    
    test_mindwave_connection(duration=10)
