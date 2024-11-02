import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
from IPython.display import HTML



def color_shape_sensitivity_hist(color_importance_norm, gaussian_sensitivity_norm, color_threshold, gaussian_threshold, num_bins=20):
    if color_importance_norm.device != torch.device('cpu'):
        color_importance_norm = color_importance_norm.cpu().numpy()

    if gaussian_sensitivity_norm.device != torch.device('cpu'):
        gaussian_sensitivity_norm = gaussian_sensitivity_norm.cpu().numpy()

    # Plotting
    _, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Color sensitivity histogram
    axes[0].hist(
        color_importance_norm,
        bins=num_bins,
        color="#1f77b4",
        density=True
    )
    axes[0].axvline(color_threshold, color='red', linestyle='--', label=f'Threshold ({color_threshold})')
    axes[0].set_title("Color Sensitivity Distribution")
    axes[0].set_xlabel("Sensitivity")
    axes[0].set_ylabel("Density")
    # axes[0].set_yscale("log")

    # Shape sensitivity histogram
    axes[1].hist(
        gaussian_sensitivity_norm,
        bins=num_bins,
        color="#ff7f0e",
        density=True
    )
    axes[1].axvline(gaussian_threshold, color='red', linestyle='--', label=f'Threshold ({gaussian_threshold})')
    axes[1].set_title("Shape Sensitivity Distribution")
    axes[1].set_xlabel("Sensitivity")

    plt.tight_layout()
    plt.show()



def scatterplot_prune_gaussians(pos_prune, pos_keep, non_prune_mask):
    fig = go.Figure()

    # Scatter plot for Gaussians that will be kept
    fig.add_trace(go.Scatter3d(
        x=pos_prune[:, 0],
        y=pos_prune[:, 1],
        z=pos_prune[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='red',    
            opacity=0.5,
        ),
        name="Pruned Gaussians"
    ))

    # Scatter plot for Gaussians that will be pruned
    fig.add_trace(go.Scatter3d(
        x=pos_keep[:, 0],
        y=pos_keep[:, 1],
        z=pos_keep[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.7,
        ),
        name="Kept Gaussians"
    ))

    prune_percentage = (1 - non_prune_mask.float().mean()) * 100

    title = f"Pruning {prune_percentage:.2f}% of Gaussians"
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        legend=dict(x=1, y=0.9)
    )

    fig.show()



def plot_features_pca(features=[], label="Features"):
    if features.device != torch.device('cpu'):
        features = features.cpu().numpy()

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], s=1, alpha=0.5, color='blue', label=label)
    
    plt.title("")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()



def plot_features_3d(features, title="3D Features", elev=30, azim=120):
    if features.device != torch.device('cpu'):
        features = features.cpu().numpy()

    # Extract each component for the 3D plot
    x = features[:, 0]
    y = features[:, 1]
    z = features[:, 2]

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, alpha=0.5, color='green', label="Scaling Features")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    ax.view_init(elev=elev, azim=azim)

    plt.show()



def animate_feature_clustering(features, centroids_history, title="Features Clustering Animation"):
    if features.device != torch.device('cpu'):
        features = features.cpu().numpy()

    # Reduce features to 2D for plotting
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of features (fixed throughout the animation)
    ax.scatter(features_2d[:, 0], features_2d[:, 1], s=1, alpha=0.5, color='blue', label="Features")
    
    # Initial scatter plot for centroids (updated in each frame)
    centroids_2d = pca.transform(centroids_history[0])  # Initial centroids
    centroids_plot, = ax.plot(centroids_2d[:, 0], centroids_2d[:, 1], 'ro', markersize=1, label="Centroids")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()

    def update(frame):
        # Update centroids plot in each frame
        centroids_2d = pca.transform(centroids_history[frame])
        centroids_plot.set_data(centroids_2d[:, 0], centroids_2d[:, 1])
        ax.set_title(f"{title} - Iteration {frame + 1}")
        return centroids_plot,

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(centroids_history), interval=500, blit=True)
    # plt.show()
    display(HTML(ani.to_jshtml()))

    return ani



def animate_feature_clustering_3d(features, centroids_history, title="3D Features Clustering Animation", elev=30, azim=120):
    # Ensure features are on the CPU and convert to numpy if needed
    if features.device != torch.device('cpu'):
        features = features.cpu().numpy()

    # Extract each component for the 3D plot
    x = features[:, 0]
    y = features[:, 1]
    z = features[:, 2]
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the initial features as a static background
    ax.scatter(x, y, z, s=1, alpha=0.5, color='green', label="Scaling Features")
    
    # Set initial view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Initialize centroid plot (updated each frame)
    centroids_3d = centroids_history[0]
    centroids_plot = ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], 
                                c='red', s=20, label="Centroids")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    def update(frame):
        # Update centroids plot for the current frame
        centroids_3d = centroids_history[frame]
        centroids_plot._offsets3d = (centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2])
        ax.set_title(f"{title} - Iteration {frame + 1}")
        return centroids_plot,

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(centroids_history), interval=500, blit=True)
    display(HTML(ani.to_jshtml()))

    return ani



def plot_features_and_compressed(features, compressed_features, title="Original and Compressed Features"):
    if features.device != torch.device('cpu'):
        features = features.cpu().numpy()

    if compressed_features.device != torch.device('cpu'):
        compressed_features = compressed_features.cpu().numpy()

    # Reduce features to 2D using PCA for visualization
    pca = PCA(n_components=2)
    all_features_2d = pca.fit_transform(features)
    compressed_features_2d = pca.transform(compressed_features)

    # Plot all features and compressed features
    plt.figure(figsize=(10, 8))
    plt.scatter(all_features_2d[:, 0], all_features_2d[:, 1], s=1, alpha=0.5, color='blue', label="All Features")
    plt.scatter(compressed_features_2d[:, 0], compressed_features_2d[:, 1], s=1, color='red', label="Compressed Features")
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()



def plot_features_and_compressed_3d(features, compressed_features, title="Original and Compressed Features 3D", elev=30, azim=120):
    if features.device != torch.device('cpu'):
        features = features.cpu().numpy()

    if compressed_features.device != torch.device('cpu'):
        compressed_features = compressed_features.cpu().numpy()

    # Extract each component for the 3D plot
    x = features[:, 0]
    y = features[:, 1]
    z = features[:, 2]

    c_x = compressed_features[:, 0]
    c_y = compressed_features[:, 1]
    c_z = compressed_features[:, 2]

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, alpha=0.1, color='blue', label="Scaling Features")
    ax.scatter(c_x, c_y, c_z, s=1, alpha=0.1, color='red', label="Scaling Features")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    ax.view_init(elev=elev, azim=azim)

    plt.show()



def plot_error_curve(errors=[]):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label='Quantization Error')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Quantization Error')
    plt.title('Quantization Error Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()



def animate_training_renders(rendered_images):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initial image display
    img_display = ax.imshow(rendered_images[0])
    ax.set_title("Rendered Image Animation")

    def update(frame):
        img_display.set_data(rendered_images[frame])
        ax.set_title(f"Rendered Image - Frame {frame + 1}")
        return img_display,

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(rendered_images), interval=500, blit=True)
    
    display(HTML(ani.to_jshtml()))
    
    return ani



def draw_ground_truth_image(gt):
    plt.figure(figsize=(8, 6))
    plt.imshow(gt)
    plt.title("Corresponding Ground Truth Image")
    plt.axis('off')
    plt.show()



def plot_finetune_losses(losses=[], window_size=100):
    plt.figure(figsize=(10, 6))
    
    # Plot the original loss values
    plt.plot(losses, label='Finetuning loss', alpha=0.5)
    
    # Calculate a moving average for the trend line
    if len(losses) >= window_size:
        moving_average = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        trend_iterations = range(len(moving_average))
        plt.plot(trend_iterations, moving_average, label='Trend line (Moving Average)', color='orange', linewidth=2)
    
    # Labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Finetuning loss')
    plt.title('Finetuning loss Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()



def visualise_timings(timings=[]):
    plt.figure(figsize=(14, 6))
    timing_keys = list(timings.keys())
    timing_values = list(timings.values())
    plt.bar(timing_keys, timing_values)
    plt.ylabel("Time (seconds)")
    plt.title("Time Spent on Each Step")

    plt.tight_layout()
    plt.show()



def visualise_storage_metrics(sizes=[]):
    labels = ['Input Model', 'Compressed Model']

    # Plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, sizes, color=['skyblue', 'orange'])
    plt.ylabel("Size (MB)")
    plt.title("Comparison of Input Model and Compressed Model Sizes")

    # Left bar
    bar, size = bars[0], sizes[0]
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval - 20, f"{size:.2f} MB", ha='center', va='bottom', fontsize=10)

    # Right bar
    bar, size = bars[1], sizes[1]
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{size:.2f} MB", ha='center', va='bottom', fontsize=10)

    plt.show()