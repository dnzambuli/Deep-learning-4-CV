def organize_balanced_clean_data(root_directories, clean_data_path="/content/Clean_Data", split_ratio=(0.7, 0.2, 0.1)):
    """
    Organizes images from cropped folders into Clean_Data with a balanced dataset.

    Args:
        root_directories (list): List of primary folder directories (e.g., ["Superficial-Intermediate", "Parabasal", ...]).
        clean_data_path (str): Destination folder for structured data.
        split_ratio (tuple): Train, validation, and test split percentages.

    Ensures:
        - The dataset is **balanced** by limiting the dominant classes to match the minority class.
        - Each dataset type (Training, Validation, Test) gets an equal proportion of images.
    """

    # Create Clean_Data directory and subfolders
    dataset_types = ["Training_Data", "Validation_Data", "Test_Data"]
    for dataset in dataset_types:
        dataset_path = os.path.join(clean_data_path, dataset)
        os.makedirs(os.path.join(dataset_path, "Images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "Masks"), exist_ok=True)

    # Count available cropped images per class
    cropped_counts = count_cropped_images(root_directories)

    # Determine the **minimum** number of images among all cell types (for balancing)
    min_images = min(cropped_counts.values())

    print(f"📊 Balancing dataset to {min_images} images per class to prevent bias.")

    # Store label info
    label_files = {dataset: [] for dataset in dataset_types}

    for root_dir in root_directories:
        cropped_folder = os.path.join(root_dir, f"im_{root_dir}", "CROPPED")

        if not os.path.exists(cropped_folder):
            print(f"⚠️ Skipping {root_dir} - No Cropped folder found.")
            continue

        # Get all images in the Cropped folder, ensuring we only use `xxx_xx.bmp` format
        images = [f for f in os.listdir(cropped_folder) if f.endswith(".bmp") and "_" in f and f.split("_")[0].isdigit()]

        # Randomly sample `min_images` to balance the dataset
        random.shuffle(images)
        balanced_images = images[:min_images]  # Ensures all classes have the same number of images

        # Split dataset
        total = len(balanced_images)
        train_split = int(split_ratio[0] * total)
        val_split = int(split_ratio[1] * total)

        train_files = balanced_images[:train_split]
        val_files = balanced_images[train_split:train_split + val_split]
        test_files = balanced_images[train_split + val_split:]

        # Assign each set to corresponding directory
        dataset_mapping = {
            "Training_Data": train_files,
            "Validation_Data": val_files,
            "Test_Data": test_files
        }

        for dataset, files in dataset_mapping.items():
            dataset_path = os.path.join(clean_data_path, dataset, "Images")
            mask_path = os.path.join(clean_data_path, dataset, "Masks")

            for img in files:
                src_path = os.path.join(cropped_folder, img)
                dst_path = os.path.join(dataset_path, img)
                shutil.copy(src_path, dst_path)  # Copy image to new location

                nuc_dat = src_path.replace(".bmp", "_nuc.dat")
                cyt_dat = src_path.replace(".bmp", "_cyt.dat")
                nuc_mask = os.path.join(mask_path, img.replace(".bmp", "_nuc.bmp"))
                cyt_mask = os.path.join(mask_path, img.replace(".bmp", "_cyt.bmp"))

                create_mask_from_dat(src_path, nuc_dat, nuc_mask)
                create_mask_from_dat(src_path, cyt_dat, cyt_mask)

                # Store label info (format: "image_name label")
                label_files[dataset].append(f"{img} {root_dir}")

    # Write Labels.txt for each dataset
    for dataset in dataset_types:
        label_txt_path = os.path.join(clean_data_path, dataset, "Labels.txt")
        with open(label_txt_path, "w") as f:
            f.write("\n".join(label_files[dataset]))

    print("✅ Data Organization Complete!")
