import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, Input, GlobalAveragePooling2D, Dense, Reshape, Dropout, Concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
from coco_parser import COCOParser
import matplotlib.patches as patches

# Get class information from the dataset
def get_class_info(annotation_file):
    coco = COCOParser(annotation_file, "")
    categories = {}
    for cat_id, cat_info in coco.cat_dict.items():
        categories[cat_id] = cat_info['name']
    
    num_classes = len(categories)
    class_names = [categories[i] for i in sorted(categories.keys())]
    return num_classes, class_names, categories

# SolarPanelDataset class
class SolarPanelDataset(tf.data.Dataset):
    def __new__(cls, images_dir, annotation_file, num_classes, categories):
        coco = COCOParser(annotation_file, images_dir)
        img_ids = coco.get_imgIds()
        image_paths = []
        bboxes = []
        labels = []
        label_names = []
        
        # Minimum class ID to adjust indexing
        min_class_id = min(categories.keys())
        
        for img_id in img_ids:
            img_info = coco.im_dict[img_id]
            filename = img_info['file_name']
            image_path = str(pathlib.Path(images_dir) / filename)
            image_paths.append(image_path)
            
            anns = coco.annIm_dict[img_id]
            img_bboxes = []
            img_labels = []
            img_label_names = []
            
            for ann in anns:
                x, y, width, height = ann['bbox']
                x_min = x
                y_min = y
                x_max = x + width
                y_max = y + height
                img_bboxes.append([x_min, y_min, x_max, y_max])
                
                # Class id and name
                class_id = ann["category_id"]
                class_name = coco.load_cats(class_id)[0]["name"]
                img_labels.append(class_id)
                img_label_names.append(class_name)

            bboxes.append(np.array(img_bboxes, dtype=np.float32))
            labels.append(np.array(img_labels, dtype=np.int32))
            label_names.append(np.array(img_label_names, dtype=object))

        # Generator function
        def generator():
            for img_path, bbox, label, label_name in zip(image_paths, bboxes, labels, label_names):
                img = tf.io.read_file(img_path)
                img = tf.image.decode_jpeg(img, channels=3)
                original_shape = tf.shape(img)[:2]
                img = tf.image.resize(img, (224, 224))
                img = tf.cast(img, tf.float32) / 255.0
                
                # Scale bounding boxes
                scale_y = 224.0 / tf.cast(original_shape[0], tf.float32)
                scale_x = 224.0 / tf.cast(original_shape[1], tf.float32)
                bbox = tf.convert_to_tensor(bbox)
                bbox_scaled = bbox * [scale_x, scale_y, scale_x, scale_y]
                
                # Create grid targets
                grid_size = 7
                cell_size = 224 / grid_size
                grid_targets = tf.zeros((grid_size, grid_size, 5 + num_classes))
                
                for i in range(tf.shape(bbox_scaled)[0]):
                    x_min, y_min, x_max, y_max = bbox_scaled[i]
                    box_center_x = (x_min + x_max) / 2
                    box_center_y = (y_min + y_max) / 2
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    
                    # Calculate grid cell
                    grid_x = tf.cast(box_center_x // cell_size, tf.int32)
                    grid_y = tf.cast(box_center_y // cell_size, tf.int32)
                    
                    # Ensure within grid bounds
                    grid_x = tf.minimum(grid_x, grid_size - 1)
                    grid_y = tf.minimum(grid_y, grid_size - 1)
                    
                    # Relative to grid cell
                    cell_x = (box_center_x - grid_x * cell_size) / cell_size
                    cell_y = (box_center_y - grid_y * cell_size) / cell_size
                    cell_w = box_width / 224.0
                    cell_h = box_height / 224.0
                    
                    # Adjust class ID to be 0-indexed
                    class_id = label[i]
                    adjusted_class_id = class_id - min_class_id
                    one_hot = tf.one_hot(adjusted_class_id, num_classes)
                    
                    # Update grid cell
                    box_info = tf.concat([[1.0, cell_x, cell_y, cell_w, cell_h], one_hot], axis=0)
                    indices = tf.constant([[grid_y, grid_x]], dtype=tf.int32)
                    grid_targets = tf.tensor_scatter_nd_update(
                        grid_targets,
                        indices,
                        tf.reshape(box_info, [1, 5 + num_classes])
                    )
                
                # Return only the image and grid_targets for model training
                yield img, grid_targets

        # Output signature - simplified to match model expectation
        output_signature = (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(7, 7, 5 + num_classes), dtype=tf.float32)
        )
        
        return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

# Additional dataset for visualization (keeps all outputs)
class SolarPanelVisualizationDataset(tf.data.Dataset):
    def __new__(cls, images_dir, annotation_file, num_classes, categories):
        coco = COCOParser(annotation_file, images_dir)
        img_ids = coco.get_imgIds()
        image_paths = []
        bboxes = []
        labels = []
        label_names = []
        
        # Minimum class ID to adjust indexing
        min_class_id = min(categories.keys())
        
        for img_id in img_ids:
            img_info = coco.im_dict[img_id]
            filename = img_info['file_name']
            image_path = str(pathlib.Path(images_dir) / filename)
            image_paths.append(image_path)
            
            anns = coco.annIm_dict[img_id]
            img_bboxes = []
            img_labels = []
            img_label_names = []
            
            for ann in anns:
                x, y, width, height = ann['bbox']
                x_min = x
                y_min = y
                x_max = x + width
                y_max = y + height
                img_bboxes.append([x_min, y_min, x_max, y_max])
                
                # Class id and name
                class_id = ann["category_id"]
                class_name = coco.load_cats(class_id)[0]["name"]
                img_labels.append(class_id)
                img_label_names.append(class_name)

            bboxes.append(np.array(img_bboxes, dtype=np.float32))
            labels.append(np.array(img_labels, dtype=np.int32))
            label_names.append(np.array(img_label_names, dtype=object))

        # Generator function
        def generator():
            for img_path, bbox, label, label_name in zip(image_paths, bboxes, labels, label_names):
                img = tf.io.read_file(img_path)
                img = tf.image.decode_jpeg(img, channels=3)
                original_shape = tf.shape(img)[:2]
                img = tf.image.resize(img, (224, 224))
                img = tf.cast(img, tf.float32) / 255.0
                
                # Scale bounding boxes
                scale_y = 224.0 / tf.cast(original_shape[0], tf.float32)
                scale_x = 224.0 / tf.cast(original_shape[1], tf.float32)
                bbox = tf.convert_to_tensor(bbox)
                bbox_scaled = bbox * [scale_x, scale_y, scale_x, scale_y]
                
                # Create grid targets
                grid_size = 7
                cell_size = 224.0 / grid_size  # Changed to 224.0
                grid_targets = tf.zeros((grid_size, grid_size, 5 + num_classes))
                
                for i in range(tf.shape(bbox_scaled)[0]):
                    x_min, y_min, x_max, y_max = bbox_scaled[i]
                    box_center_x = (x_min + x_max) / 2
                    box_center_y = (y_min + y_max) / 2
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    
                    # Calculate grid cell
                    grid_x = tf.cast(box_center_x // cell_size, tf.int32)
                    grid_y = tf.cast(box_center_y // cell_size, tf.int32)
                    
                    # Ensure within grid bounds
                    grid_x = tf.minimum(grid_x, grid_size - 1)
                    grid_y = tf.minimum(grid_y, grid_size - 1)
                    
                    # Relative to grid cell - cast grid coordinates to float32
                    cell_x = (box_center_x - tf.cast(grid_x, tf.float32) * cell_size) / cell_size
                    cell_y = (box_center_y - tf.cast(grid_y, tf.float32) * cell_size) / cell_size
                    cell_w = box_width / 224.0
                    cell_h = box_height / 224.0
                    
                    # Adjust class ID to be 0-indexed
                    class_id = label[i]
                    adjusted_class_id = class_id - min_class_id
                    one_hot = tf.one_hot(adjusted_class_id, num_classes)
                    
                    # Update grid cell
                    box_info = tf.concat([[1.0, cell_x, cell_y, cell_w, cell_h], one_hot], axis=0)
                    indices = tf.stack([[grid_y, grid_x]])  # Changed from tf.constant
                    grid_targets = tf.tensor_scatter_nd_update(
                        grid_targets,
                        indices,
                        tf.reshape(box_info, [1, 5 + num_classes])
                    )
                
                # Return all data for visualization
                yield img, {
                    'grid_targets': grid_targets,
                    'boxes': bbox_scaled,
                    'labels': label,
                    'label_names': label_name
                }

        # Output signature for visualization
        output_signature = (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            {
                'grid_targets': tf.TensorSpec(shape=(7, 7, 5 + num_classes), dtype=tf.float32),
                'boxes': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'label_names': tf.TensorSpec(shape=(None,), dtype=tf.string),
            }
        )
        
        return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

# Model creation function
def create_solar_panel_defect_detector(num_classes):
    # Use ResNet50 backbone
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add detection head
    x = base_model.output
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    
    # Resize to match the grid size
    x = tf.keras.layers.Conv2D(5 + num_classes, (1, 1), padding='same')(x)
    x = tf.keras.layers.Reshape((7, 7, 5 + num_classes))(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=x)
    
    return model

# Loss function
def yolo_loss(y_true, y_pred):
    # Object confidence loss
    obj_mask = y_true[:, :, :, 0:1]
    conf_loss = tf.reduce_sum(tf.square(obj_mask * (y_true[:, :, :, 0:1] - y_pred[:, :, :, 0:1])))
    
    # No-object confidence loss
    noobj_mask = 1.0 - obj_mask
    noobj_loss = 0.5 * tf.reduce_sum(tf.square(noobj_mask * y_pred[:, :, :, 0:1]))
    
    # Coordinate loss
    xy_loss = tf.reduce_sum(obj_mask * tf.square(y_true[:, :, :, 1:3] - y_pred[:, :, :, 1:3]))
    
    # Width height loss
    wh_loss = tf.reduce_sum(obj_mask * tf.square(
        tf.sqrt(y_true[:, :, :, 3:5]) - tf.sqrt(tf.maximum(y_pred[:, :, :, 3:5], 1e-10))
    ))
    
    # Class prediction loss
    class_loss = tf.reduce_sum(obj_mask * tf.square(y_true[:, :, :, 5:] - y_pred[:, :, :, 5:]))
    
    # Total loss
    total_loss = conf_loss + noobj_loss + xy_loss + wh_loss + class_loss
    
    return total_loss

# Prepare datasets
def prepare_datasets():
    # Get class info first
    train_annotation_file = "Solar Panel Fault Dataset.v1i.coco/train/_annotations.coco.json"
    num_classes, class_names, categories = get_class_info(train_annotation_file)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create datasets with class info
    train_ds = SolarPanelDataset(
        images_dir="Solar Panel Fault Dataset.v1i.coco/train",
        annotation_file=train_annotation_file,
        num_classes=num_classes,
        categories=categories
    )
    
    val_ds = SolarPanelDataset(
        images_dir="Solar Panel Fault Dataset.v1i.coco/valid",
        annotation_file="Solar Panel Fault Dataset.v1i.coco/valid/_annotations.coco.json",
        num_classes=num_classes,
        categories=categories
    )
    
    # Create visualization dataset
    vis_ds = SolarPanelVisualizationDataset(
        images_dir="Solar Panel Fault Dataset.v1i.coco/test",
        annotation_file="Solar Panel Fault Dataset.v1i.coco/test/_annotations.coco.json",
        num_classes=num_classes,
        categories=categories
    )
    
    # Batch and prefetch
    train_ds = train_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    vis_ds = vis_ds.batch(4).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, vis_ds, num_classes, class_names

# Non-max suppression
def non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5):
    selected_indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=100,
        iou_threshold=iou_threshold
    )
    
    selected_boxes = tf.gather(boxes, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)
    selected_classes = tf.gather(class_ids, selected_indices)
    
    return selected_boxes, selected_scores, selected_classes

# Visualization function
def visualize_detections(model, dataset, class_names, num_images=4):
    for images, targets in dataset.take(1):
        # Make predictions
        predictions = model.predict(images)
        
        # Plot results
        fig, axs = plt.subplots(min(num_images, len(images)), 2, figsize=(16, 4 * min(num_images, len(images))))
        
        for i in range(min(num_images, len(images))):
            # Ground truth
            if num_images == 1:
                ax_gt = axs[0]
                ax_pred = axs[1]
            else:
                ax_gt = axs[i, 0]
                ax_pred = axs[i, 1]
                
            ax_gt.imshow(images[i])
            ax_gt.set_title("Ground Truth")
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
            ax_pred.set_title("Predictions")
            ax_pred.axis('off')
            
            # Process prediction grid
            pred_grid = predictions[i]
            grid_size = 7
            cell_size = 224 / grid_size
            
            confidence_threshold = 0.3
            
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
                    ax_pred.add_patch(rect)
                    ax_pred.text(x_min, y_min - 5, 
                                f"{class_names[class_id]} ({score:.2f})",
                                fontsize=8, color='black',
                                bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
        
        plt.tight_layout()
        plt.show()
        break

# Train model function
def train_model(model, train_ds, val_ds, num_epochs=20):
    # Create custom training loop
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Setup callbacks
    checkpoint_filepath = 'solar_panel_defect_model/model_checkpoint'
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=yolo_loss
    )
    
    # Train model
    history = model.fit(
        train_ds,  # Now train_ds returns (image, grid_targets)
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ]
    )
    
    return model, history

# Single image prediction
def predict_image(model, image_path, class_names):
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img_resized = tf.image.resize(img, (224, 224))
    img_normalized = tf.cast(img_resized, tf.float32) / 255.0
    img_batch = tf.expand_dims(img_normalized, 0)
    
    # Make prediction
    prediction = model.predict(img_batch)[0]
    
    # Process prediction grid
    grid_size = 7
    cell_size = 224 / grid_size
    
    # Lists for detections
    detected_boxes = []
    detected_scores = []
    detected_classes = []
    
    confidence_threshold = 0.3
    
    for row in range(grid_size):
        for col in range(grid_size):
            cell_pred = prediction[row, col]
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
                
                detected_boxes.append([x_min, y_min, x_max, y_max])
                detected_scores.append(confidence)
                detected_classes.append(class_id)
    
    # Apply NMS
    if detected_boxes:
        boxes_tensor = tf.convert_to_tensor(detected_boxes, dtype=tf.float32)
        scores_tensor = tf.convert_to_tensor(detected_scores, dtype=tf.float32)
        classes_tensor = tf.convert_to_tensor(detected_classes, dtype=tf.int32)
        
        boxes_nms, scores_nms, classes_nms = non_max_suppression(
            boxes_tensor, scores_tensor, classes_tensor
        )
        
        # Visualize result
        plt.figure(figsize=(10, 8))
        plt.imshow(img_normalized)
        plt.axis('off')
        
        for box, score, class_id in zip(boxes_nms.numpy(), scores_nms.numpy(), classes_nms.numpy()):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height,
                                   linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x_min, y_min - 5, 
                    f"{class_names[class_id]} ({score:.2f})",
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
        
        plt.title('Solar Panel Defect Detection Results')
        plt.show()
        
        return boxes_nms, scores_nms, classes_nms
    else:
        print("No defects detected in the image.")
        return None, None, None

# Main execution function
def main():
    # Prepare datasets
    train_ds, val_ds, vis_ds, num_classes, class_names = prepare_datasets()
    
    # Create model - make sure output shape is correct
    model = create_solar_panel_defect_detector(num_classes)
    model.summary()
    
    # Train model
    trained_model, history = train_model(model, train_ds, val_ds)
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Visualize detections
    visualize_detections(trained_model, vis_ds, class_names)
    
    # Save model
    trained_model.save('solar_panel_defect_model/final_model')
    print("Model saved to 'solar_panel_defect_model/final_model'")

if __name__ == "__main__":
    main()