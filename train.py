from ultralytics import YOLO

def main():
    # Define the path to the data configuration file
    data_config_path = 'data.yaml'

    # Create a model instance using a pre-trained model
    model = YOLO('yolov8n.pt')  # Using YOLOv8n pre-trained model

    # Start training
    model.train(
        data=data_config_path,
        epochs=150,
        batch=16,
        imgsz=640,
        device=0,  # Use the first GPU
        project='model_ten_modify',   ###
        name='model_collections',
        exist_ok=True
    )
    

if __name__ == '__main__':
    main()
