import time, torch
from djitellopy import Tello
from tello_obstacle_detection.depth_pipeline.midas_trigger import init_depth_model
from drone_navigator.drone_navigator import execute_simple_route
from tello_obstacle_detection.drone_navigator.drone_navigator import DroneKeepAlive

def make_depth_context():
    """Create and return the MiDaS depth_context dict."""
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,transform,net_w,net_h=init_depth_model(
        device,
        "src/tello_obstacle_detection/weights/dpt_swin2_large_384.pt",
        model_type="dpt_swin2_large_384",
        optimize=False,
    )
    return {"device":device,"model":model,"model_type":"dpt_swin2_large_384","transform":transform,"net_w":net_w,"net_h":net_h,"optimize":False}

def main():
    WIDTH=3.0;LENGTH=5.0;SPACING=0.5;GOAL=(2.5,4.0);ALTITUDE=1.0
    drone=Tello();ka=None
    try:
        drone.connect();drone.streamon();time.sleep(2);drone.takeoff();time.sleep(3)
        ka=DroneKeepAlive(drone,interval=10);ka.start()
        depth_ctx=make_depth_context()
        execute_simple_route(drone,WIDTH,LENGTH,SPACING,GOAL,ALTITUDE,depth_ctx,depth_callback=None)
    finally:
        if ka: ka.stop()
        try: drone.land();time.sleep(3);drone.streamoff()
        except Exception as e: print(f"[Cleanup] {e}")

if __name__=="__main__":
    main()
