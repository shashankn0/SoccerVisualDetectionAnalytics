# comprehensive model training pipeline for soccer detection
import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import shutil

class SoccerModelTrainer:
    """comprehensive training system for soccer detection models."""
    
    def __init__(self):
        self.dataset_paths = {
            'players': 'soccer-analytics-yolo/datasets/players',
            'balls': 'soccer-analytics-yolo/datasets/balls', 
            'field': 'soccer-analytics-yolo/datasets/field'
        }
        self.model_paths = {
            'players': 'models/player.pt',
            'balls': 'models/ball.pt',
            'field': 'models/field.pt'
        }
        self.ensure_directories()
    
    def ensure_directories(self):
        """ensure all required directories exist."""
        directories = [
            'models',
            'runs/detect',
            'datasets',
            'training_logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"+ ensured {directory}/ directory exists")
    
    def prepare_dataset_config(self, dataset_type, dataset_path):
        """prepare yolo dataset configuration."""
        if not os.path.exists(dataset_path):
            print(f"- dataset path not found: {dataset_path}")
            return None
        
        # find data.yaml file
        data_yaml = os.path.join(dataset_path, 'data.yaml')
        if not os.path.exists(data_yaml):
            # create basic data.yaml
            config = {
                'train': os.path.join(dataset_path, 'train/images'),
                'val': os.path.join(dataset_path, 'valid/images'),
                'test': os.path.join(dataset_path, 'test/images'),
                'nc': 1,  # number of classes
                'names': [dataset_type[:-1]]  # remove 's' from plural
            }
            
            with open(data_yaml, 'w') as f:
                yaml.dump(config, f)
            
            print(f"+ created data.yaml for {dataset_type}")
        
        return data_yaml
    
    def train_single_model(self, model_type, epochs=100, img_size=640, batch_size=16):
        """train a single yolo model."""
        print(f"\n{'='*50}")
        print(f"training {model_type} model")
        print(f"{'='*50}")
        
        dataset_path = self.dataset_paths.get(model_type)
        if not dataset_path or not os.path.exists(dataset_path):
            print(f"- dataset not found for {model_type}: {dataset_path}")
            return None
        
        # prepare dataset config
        data_yaml = self.prepare_dataset_config(model_type, dataset_path)
        if not data_yaml:
            return None
        
        # initialize model
        model = YOLO('yolov8n.pt')  # start with pretrained model
        
        # training parameters
        training_params = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'name': f'{model_type}_model',
            'project': 'runs/detect',
            'exist_ok': True,
            'patience': 50,  # early stopping
            'save_period': 10,  # save every 10 epochs
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 4 if torch.cuda.is_available() else 2,
            'lr0': 0.01,  # learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,  # box loss gain
            'cls': 0.5,  # classification loss gain
            'dfl': 1.5,  # distribution focal loss gain
            'pose': 12.0,  # pose loss gain (for keypoints)
            'kobj': 1.0,  # keypoint objectness loss gain
            'label_smoothing': 0.0,
            'nbs': 64,  # nominal batch size
            'hsv_h': 0.015,  # hue augmentation
            'hsv_s': 0.7,    # saturation augmentation
            'hsv_v': 0.4,    # value augmentation
            'degrees': 0.0,   # rotation augmentation
            'translate': 0.1,  # translation augmentation
            'scale': 0.5,     # scale augmentation
            'shear': 0.0,     # shear augmentation
            'perspective': 0.0, # perspective augmentation
            'flipud': 0.0,    # vertical flip augmentation
            'fliplr': 0.5,     # horizontal flip augmentation
            'mosaic': 1.0,     # mosaic augmentation
            'mixup': 0.0,       # mixup augmentation
            'copy_paste': 0.0,   # copy-paste augmentation
        }
        
        try:
            # train model
            results = model.train(**training_params)
            
            # get best model path
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            
            # copy to models directory
            target_path = self.model_paths[model_type]
            shutil.copy2(best_model_path, target_path)
            
            print(f"+ training completed for {model_type}")
            print(f"+ best model saved to: {target_path}")
            print(f"+ training metrics:")
            print(f"  - map50: {results.results_dict.get('metrics/mAP50', 'N/A')}")
            print(f"  - map50-95: {results.results_dict.get('metrics/mAP50-95', 'N/A')}")
            
            return target_path
            
        except Exception as e:
            print(f"- training failed for {model_type}: {e}")
            return None
    
    def train_all_models(self, epochs=100):
        """train all soccer detection models."""
        print("starting comprehensive model training...")
        print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        trained_models = {}
        
        for model_type in ['players', 'balls', 'field']:
            model_path = self.train_single_model(model_type, epochs)
            if model_path:
                trained_models[model_type] = model_path
        
        # summary
        print(f"\n{'='*50}")
        print("training summary:")
        print(f"{'='*50}")
        
        for model_type, path in trained_models.items():
            status = "+" if path else "-"
            print(f"{status} {model_type}: {path}")
        
        print(f"\ntrained {len(trained_models)}/3 models successfully")
        
        return trained_models
    
    def validate_model(self, model_path, dataset_type):
        """validate trained model performance."""
        print(f"\nvalidating {dataset_type} model...")
        
        if not os.path.exists(model_path):
            print(f"- model not found: {model_path}")
            return None
        
        try:
            # load model
            model = YOLO(model_path)
            
            # validation dataset path
            val_path = os.path.join(self.dataset_paths[dataset_type], 'valid/images')
            if not os.path.exists(val_path):
                print(f"- validation dataset not found: {val_path}")
                return None
            
            # run validation
            results = model.val(data=val_path)
            
            print(f"+ validation completed for {dataset_type}")
            print(f"+ validation metrics:")
            print(f"  - map50: {results.box.map50:.3f}")
            print(f"  - map50-95: {results.box.map:.3f}")
            print(f"  - precision: {results.box.mp:.3f}")
            print(f"  - recall: {results.box.mr:.3f}")
            
            return results
            
        except Exception as e:
            print(f"- validation failed: {e}")
            return None
    
    def validate_all_models(self):
        """validate all trained models."""
        print("validating all trained models...")
        
        validation_results = {}
        
        for model_type in ['players', 'balls', 'field']:
            model_path = self.model_paths[model_type]
            if os.path.exists(model_path):
                results = self.validate_model(model_path, model_type)
                validation_results[model_type] = results
            else:
                print(f"- model not found for {model_type}: {model_path}")
        
        return validation_results
    
    def create_training_report(self, validation_results):
        """create comprehensive training report."""
        report_path = 'training_logs/model_performance_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("soccer detection models - training report\n")
            f.write("=" * 50 + "\n\n")
            
            for model_type, results in validation_results.items():
                f.write(f"{model_type.upper()} model:\n")
                f.write("-" * 20 + "\n")
                
                if results:
                    f.write(f"map50: {results.box.map50:.3f}\n")
                    f.write(f"map50-95: {results.box.map:.3f}\n")
                    f.write(f"precision: {results.box.mp:.3f}\n")
                    f.write(f"recall: {results.box.mr:.3f}\n")
                else:
                    f.write("validation failed\n")
                
                f.write("\n")
        
        print(f"+ training report saved to: {report_path}")

def main():
    """main training function."""
    trainer = SoccerModelTrainer()
    
    # check available datasets
    print("checking available datasets...")
    for model_type, path in trainer.dataset_paths.items():
        if os.path.exists(path):
            print(f"+ {model_type} dataset: {path}")
        else:
            print(f"- {model_type} dataset not found: {path}")
    
    # train all models
    print("\nstarting training pipeline...")
    trained_models = trainer.train_all_models(epochs=50)  # reduced for testing
    
    # validate models
    if trained_models:
        print("\nvalidating trained models...")
        validation_results = trainer.validate_all_models()
        
        # create report
        trainer.create_training_report(validation_results)
    
    print("\ntraining pipeline completed!")

if __name__ == "__main__":
    main()
