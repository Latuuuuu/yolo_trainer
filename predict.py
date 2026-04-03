import cv2
from ultralytics import YOLO
from pathlib import Path

source_folder = '/usr/src/imgs/2'
model = YOLO('runs/table_tennis_1280/weights/best.pt')

def main():
    for img in Path(source_folder).glob('*.jpg'):
        # predict without show=True since it only waits 1 millisecond natively
        results = model.predict(source=img, conf=0.25, verbose=False, device=0)
        print(results[0].speed)

        # Get the annotated image as a numpy array
        annotated_img = results[0].plot()
        
        # Manually display the image with OpenCV
        cv2.imshow("Detection Result", annotated_img)

        # Wait indefinitely for a key press (0 means wait forever)
        print("Press any key on the image window to continue...")
        cv2.waitKey(0)

        # Check if window is closed
        if cv2.getWindowProperty("Detection Result", cv2.WND_PROP_VISIBLE) < 1:  
            print("Window closed by user. Exiting.")
            break
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()