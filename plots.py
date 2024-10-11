import plotly
import numpy as np

def plot_latents_3d(z, labels, discrete=True, markersize=1, alpha=0.1):

    axis = plotly.graph_objects.Figure(
        layout=plotly.graph_objects.Layout(height=500,
                                            width=500))
    #
    traces = []

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
                                                    colorscale='viridis',
                                                    ),
                                                name=str(label))
    for trace in traces:
        axis.add_trace(trace)

    return axis

