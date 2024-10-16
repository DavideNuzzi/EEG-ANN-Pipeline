import plotly
import numpy as np


def plot_latents_3d(z, labels, discrete=True, show_legend=True, markersize=1, alpha=0.1, title='', figsize=(4,4)):

    axis = plotly.graph_objects.Figure(
        layout=plotly.graph_objects.Layout(height=100 * figsize[0],
                                            width=100 * figsize[1]))
    
    data = []

    if discrete:
        unique_labels = np.unique(labels)
    else:
        unique_labels = [labels]

    for label in unique_labels:

        if discrete:
            z_masked = z[labels == label, :]
        else:
            z_masked = z

        trace = plotly.graph_objects.Scatter3d(x=z_masked[:,0],
                                               y=z_masked[:,1],
                                               z=z_masked[:,2],
                                               mode="markers",
                                                marker=dict(
                                                    size=markersize,
                                                    opacity=alpha,
                                                    color=label,
                                                    colorscale='inferno'
                                                ),
                                                # line=dict(
                                                #     color=label,
                                                #     width=0.2
                                                # ),
                                                name=str(label))
        data.append(trace)
        
    for trace in data:
        axis.add_trace(trace)

    axis.update_layout(
        template='plotly_white',
        showlegend=show_legend,
        title=title,
        scene_aspectmode='cube'
    )

    
    return axis



def plot_latent_trajectories_3d(z_traj, labels, show_legend=True, linewidth=1, title='', figsize=(4,4)):

    axis = plotly.graph_objects.Figure(
        layout=plotly.graph_objects.Layout(height=100 * figsize[0],
                                            width=100 * figsize[1]))
    
    data = []
    N_lines = len(z_traj)
    
    for i in range(N_lines):
        
        z = z_traj[i]
        label = labels[i]

        trace = plotly.graph_objects.Scatter3d(x=z[:,0],
                                               y=z[:,1],
                                               z=z[:,2],
                                               mode="lines",
                                                line=dict(
                                                    color=label*1,
                                                    colorscale='jet',
                                                    width=linewidth
                                                ),
                                                name=str(label))
        data.append(trace)
        
    for trace in data:
        axis.add_trace(trace)

    axis.update_layout(
        template='plotly_white',
        showlegend=show_legend,
        title=title,
        scene_aspectmode='cube'
    )

    
    return axis

