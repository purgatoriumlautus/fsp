import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pathlib
import os
from coco_parser import COCOParser

# Import functions from your main file
from img_clas import (
    get_class_info, 
    SolarPanelVisualizationDataset, 
    yolo_loss, 
    non_max_suppression,
    predict_image
)

class ModelEvaluator:
    def __init__(self, model_path, annotation_file):
        """Initialize the model evaluator"""
        # Load the trained model
        self.model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'yolo_loss': yolo_loss}
        )
        
        # Get class information
        self.num_classes, self.class_names, self.categories = get_class_info(annotation_file)
        print(f"Loaded model with {self.num_classes} classes: {self.class_names}")
    
    def evaluate_on_test_set(self, test_images_dir, test_annotation_file):
        """Evaluate model on test dataset"""
        print("Creating test dataset...")
        test_ds = SolarPanelVisualizationDataset(
            images_dir=test_images_dir,
            annotation_file=test_annotation_file,
            num_classes=self.num_classes,
            categories=self.categories
        ).batch(1)  # Changed from .batch(4) to .batch(1)
        
        print("Visualizing detections on test set...")
        self.visualize_detections(test_ds, num_images=8)
        
        # Calculate basic metrics
        self.calculate_detection_metrics(test_ds)
    
    def visualize_detections(self, dataset, num_images=4):
        """Visualize model predictions vs ground truth"""
        images_shown = 0
        
        for images, targets in dataset:
            if images_shown >= num_images:
                break
                
            # Make predictions
            predictions = self.model.predict(images)
            
            # Calculate how many images to show from this batch
            batch_size = len(images)
            remaining_images = min(num_images - images_shown, batch_size)
            
            # Plot results
            fig, axs = plt.subplots(remaining_images, 2, figsize=(16, 4 * remaining_images))
            if remaining_images == 1:
                axs = axs.reshape(1, -1)
            
            for i in range(remaining_images):
                # Ground truth
                ax_gt = axs[i, 0]
                ax_pred = axs[i, 1]
                
                ax_gt.imshow(images[i])
                ax_gt.set_title(f"Ground Truth - Image {images_shown + i + 1}")
                ax_gt.axis('off')
                
                # Draw ground truth boxes
                for box, label_name in zip(targets['boxes'][i], targets['label_names'][i]):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = patches.Rectangle((x_min, y_min), width, height,
                                            linewidth=2, edgecolor='green', facecolor='none')
                    ax_gt.add_patch(rect)
                    ax_gt.text(x_min, y_min - 5, label_name.numpy().decode('utf-8'),
                              fontsize=8, color='black',
                              bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.2'))
                
                # Image with predictions
                ax_pred.imshow(images[i])
                ax_pred.set_title(f"Predictions - Image {images_shown + i + 1}")
                ax_pred.axis('off')
                
                # Process prediction grid
                self._draw_predictions(ax_pred, predictions[i])
            
            plt.tight_layout()
            plt.show()
            
            images_shown += remaining_images
    
    def _draw_predictions(self, ax, pred_grid, confidence_threshold=0.3):
        """Draw predictions on the given axis"""
        grid_size = 7
        cell_size = 224 / grid_size
        
        # Lists for NMS
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for row in range(grid_size):
            for col in range(grid_size):
                cell_pred = pred_grid[row, col]
                confidence = cell_pred[0]
                
                if confidence > confidence_threshold:
                    # Extract coordinates
                    cx = (col + cell_pred[1]) * cell_size
                    cy = (row + cell_pred[2]) * cell_size
                    w = cell_pred[3] * 224
                    h = cell_pred[4] * 224
                    
                    # Convert to box coordinates
                    x_min = cx - w/2
                    y_min = cy - h/2
                    x_max = x_min + w
                    y_max = y_min + h
                    
                    # Get class
                    class_probs = cell_pred[5:]
                    class_id = tf.argmax(class_probs)
                    
                    all_boxes.append([x_min, y_min, x_max, y_max])
                    all_scores.append(confidence)
                    all_classes.append(class_id)
        
        # Apply NMS
        if all_boxes:
            boxes_tensor = tf.convert_to_tensor(all_boxes, dtype=tf.float32)
            scores_tensor = tf.convert_to_tensor(all_scores, dtype=tf.float32)
            classes_tensor = tf.convert_to_tensor(all_classes, dtype=tf.int32)
            
            boxes_nms, scores_nms, classes_nms = non_max_suppression(
                boxes_tensor, scores_tensor, classes_tensor
            )
            
            # Draw prediction boxes
            for box, score, class_id in zip(boxes_nms.numpy(), scores_nms.numpy(), classes_nms.numpy()):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                rect = patches.Rectangle((x_min, y_min), width, height,
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 5, 
                        f"{self.class_names[class_id]} ({score:.2f})",
                        fontsize=8, color='black',
                        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
    
    def calculate_detection_metrics(self, test_ds, confidence_threshold=0.3):
        """Calculate basic detection metrics"""
        total_predictions = 0
        total_ground_truth = 0
        high_confidence_predictions = 0
        
        print(f"\nCalculating detection metrics with confidence threshold: {confidence_threshold}")
        
        for images, targets in test_ds.take(10):  # Evaluate on first 10 batches
            predictions = self.model.predict(images, verbose=0)
            
            for i in range(len(images)):
                # Count ground truth objects
                gt_boxes = targets['boxes'][i]
                total_ground_truth += len(gt_boxes)
                
                # Count predictions
                pred_grid = predictions[i]
                grid_size = 7
                
                for row in range(grid_size):
                    for col in range(grid_size):
                        confidence = pred_grid[row, col, 0]
                        if confidence > confidence_threshold:
                            high_confidence_predictions += 1
                        if confidence > 0.1:  # Count all reasonable predictions
                            total_predictions += 1
        
        print(f"Total ground truth objects: {total_ground_truth}")
        print(f"Total predictions (conf > 0.1): {total_predictions}")
        print(f"High confidence predictions (conf > {confidence_threshold}): {high_confidence_predictions}")
        
        if total_ground_truth > 0:
            recall_estimate = high_confidence_predictions / total_ground_truth
            print(f"Estimated recall: {recall_estimate:.3f}")
        
        return {
            'total_gt': total_ground_truth,
            'total_pred': total_predictions,
            'high_conf_pred': high_confidence_predictions
        }
    
    def test_single_image(self, image_path):
        """Test model on a single image"""
        print(f"Testing on single image: {image_path}")
        boxes, scores, classes = predict_image(self.model, image_path, self.class_names)
        
        if boxes is not None:
            print(f"Detected {len(boxes)} objects:")
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
                print(f"  {i+1}. {self.class_names[class_id]} - Confidence: {score:.3f}")
        else:
            print("No objects detected")
        
        return boxes, scores, classes
    
    def model_summary(self):
        """Display model summary and information"""
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model output shape: {self.model.output_shape}")
        
        # Count parameters
        total_params = self.model.count_params()
        print(f"Total parameters: {total_params:,}")
        
        print("\nModel architecture:")
        self.model.summary()


def main():
    """Main evaluation function"""
    # Paths to your model and data
    model_path = 'solar_panel_defect_model/model_checkpoint'
    train_annotation_file = "Solar Panel Fault Dataset.v1i.coco/train/_annotations.coco.json"
    test_images_dir = "Solar Panel Fault Dataset.v1i.coco/test"
    test_annotation_file = "Solar Panel Fault Dataset.v1i.coco/test/_annotations.coco.json"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running img_clas.py")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, train_annotation_file)
    
    # Display model information
    evaluator.model_summary()
    
    # Evaluate on test set
    evaluator.evaluate_on_test_set(test_images_dir, test_annotation_file)
    
    # Test on individual images (if you have specific images)
    # Replace with actual image paths from your test set
    sample_images = [
        "Solar Panel Fault Dataset.v1i.coco/test/sample1.jpg",  # Replace with actual paths
        "Solar Panel Fault Dataset.v1i.coco/test/sample2.jpg",
    ]
    
    for image_path in sample_images:
        if os.path.exists(image_path):
            evaluator.test_single_image(image_path)
            print("-" * 30)


if __name__ == "__main__":
    main()