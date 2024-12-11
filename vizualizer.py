import dash
from dash import html, dcc
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import os


# Reuse the helper functions from above or import them

def create_dash_app(df, output_dir):
    app = dash.Dash(__name__)

    # Create Plotly figures
    fig_reward_dist = px.histogram(df, x='reward', nbins=50, title='Reward Distribution')
    fig_reward_dist.update_layout(xaxis_title='Reward', yaxis_title='Count')

    # Apply PCA to state if needed
    state_pca_cols = [col for col in df.columns if col.startswith('state_pca')]
    if not state_pca_cols:
        state_columns = [f'state_{i}' for i in range(len(df['state'][0]))]
        df = reduce_dimensions(df, state_columns, n_components=3)
        state_pca_cols = [col for col in df.columns if col.startswith('state_pca')]

    fig_reward_vs_state = px.scatter_3d(
        df,
        x=state_pca_cols[0],
        y=state_pca_cols[1],
        z=state_pca_cols[2],
        color='reward',
        title='Reward vs. State (PCA Reduced)',
        labels={
            state_pca_cols[0]: 'State PCA 1',
            state_pca_cols[1]: 'State PCA 2',
            state_pca_cols[2]: 'State PCA 3',
            'reward': 'Reward'
        },
        opacity=0.7
    )
    fig_reward_vs_state.update_traces(marker=dict(size=3))

    # Similarly, create other figures...

    # Define the layout of the dashboard
    app.layout = html.Div(children=[
        html.H1(children='SAC Agent Training Visualizations'),

        html.Div(children='''
            Reward Distribution:
        '''),
        dcc.Graph(
            id='reward-distribution',
            figure=fig_reward_dist
        ),

        html.Div(children='''
            Reward vs. State:
        '''),
        dcc.Graph(
            id='reward-vs-state',
            figure=fig_reward_vs_state
        ),

        # Add more Graph components for other visualizations
    ])

    return app


def load_transitions(log_dir):
    transitions_path = os.path.join(log_dir, 'transitions.csv')
    df = pd.read_csv(transitions_path)

    # Parse JSON strings into lists
    df['state'] = df['state'].apply(json.loads)
    df['action'] = df['action'].apply(json.loads)
    df['next_state'] = df['next_state'].apply(json.loads)
    df['confounder'] = df['confounder'].apply(json.loads)

    # Convert lists to separate columns if needed (for states/actions with multiple dimensions)
    # For simplicity, we'll assume states and actions are 3-dimensional. Adjust accordingly.
    state_dim = len(df['state'][0])
    action_dim = len(df['action'][0])

    for i in range(state_dim):
        df[f'state_{i}'] = df['state'].apply(lambda x: x[i])

    for i in range(action_dim):
        df[f'action_{i}'] = df['action'].apply(lambda x: x[i])

    return df

def reduce_dimensions(df, columns, n_components=3):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df[columns])
    for i in range(n_components):
        df[f'{columns[0]}_pca_{i+1}'] = reduced[:, i]
    return df

def plot_reward_distribution(df, output_dir):
    fig = px.histogram(df, x='reward', nbins=50, title='Reward Distribution')
    fig.update_layout(xaxis_title='Reward', yaxis_title='Count')
    fig.write_html(os.path.join(output_dir, 'reward_distribution.html'))


def plot_reward_vs_state(df, log_dir, output_dir):
    # Identify state PCA columns
    state_pca_cols = [col for col in df.columns if col.startswith('state_pca')]
    if not state_pca_cols:
        # Apply PCA if not already done
        state_columns = [f'state_{i}' for i in range(len(df['state'][0]))]
        df = reduce_dimensions(df, state_columns, n_components=3)
        state_pca_cols = [col for col in df.columns if col.startswith('state_pca')]

    fig = px.scatter_3d(
        df,
        x=state_pca_cols[0],
        y=state_pca_cols[1],
        z=state_pca_cols[2],
        color='reward',
        title='Reward vs. State (PCA Reduced)',
        labels={
            state_pca_cols[0]: 'State PCA 1',
            state_pca_cols[1]: 'State PCA 2',
            state_pca_cols[2]: 'State PCA 3',
            'reward': 'Reward'
        },
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=3))
    fig.write_html(os.path.join(output_dir, 'reward_vs_state.html'))


def plot_action_distribution(df, output_dir):
    action_cols = [col for col in df.columns if col.startswith('action_')]
    if not action_cols:
        raise ValueError("No action columns found in the DataFrame.")

    # Create separate histograms for each action dimension
    for action_dim in action_cols:
        fig = px.histogram(df, x=action_dim, nbins=50, title=f'Action Distribution - {action_dim}')
        fig.update_layout(xaxis_title='Action Value', yaxis_title='Count')
        fig.write_html(os.path.join(output_dir, f'{action_dim}_distribution.html'))


def plot_policy(df, log_dir, output_dir):
    # Identify state PCA columns
    state_pca_cols = [col for col in df.columns if col.startswith('state_pca')]
    if not state_pca_cols:
        # Apply PCA if not already done
        state_columns = [f'state_{i}' for i in range(len(df['state'][0]))]
        df = reduce_dimensions(df, state_columns, n_components=3)
        state_pca_cols = [col for col in df.columns if col.startswith('state_pca')]

    action_cols = [col for col in df.columns if col.startswith('action_')]
    if not action_cols:
        raise ValueError("No action columns found in the DataFrame.")

    # For simplicity, plot action dimensions against first two state PCA components
    for i, action_dim in enumerate(action_cols):
        fig = px.scatter(
            df,
            x=state_pca_cols[0],
            y=state_pca_cols[1],
            color=action_dim,
            title=f'Policy Visualization - {action_dim} vs State PCA 1 & 2',
            labels={
                state_pca_cols[0]: 'State PCA 1',
                state_pca_cols[1]: 'State PCA 2',
                action_dim: f'Action {i + 1}'
            },
            opacity=0.6
        )
        fig.update_traces(marker=dict(size=3))
        fig.write_html(os.path.join(output_dir, f'policy_{action_dim}_vs_state.html'))


def plot_actions_3d(df, output_dir):
    action_cols = [col for col in df.columns if col.startswith('action_')]
    if len(action_cols) < 3:
        raise ValueError("Need at least 3 action dimensions for a 3D plot.")

    fig = px.scatter_3d(
        df,
        x=action_cols[0],
        y=action_cols[1],
        z=action_cols[2],
        color='reward',
        title='3D Action Distribution Colored by Reward',
        labels={
            action_cols[0]: 'Action Dimension 1',
            action_cols[1]: 'Action Dimension 2',
            action_cols[2]: 'Action Dimension 3',
            'reward': 'Reward'
        },
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=3))
    fig.write_html(os.path.join(output_dir, 'actions_3d_reward.html'))


def plot_advanced_3d_scatter(df, output_dir):
    # Assuming state is reduced to 3 dimensions
    state_pca_cols = [col for col in df.columns if col.startswith('state_pca')]
    action_cols = [col for col in df.columns if col.startswith('action_')]

    if len(state_pca_cols) < 3 or len(action_cols) < 3:
        raise ValueError("Ensure that state and action have at least 3 PCA components.")

    fig = go.Figure(data=[go.Scatter3d(
        x=df[state_pca_cols[0]],
        y=df[state_pca_cols[1]],
        z=df[state_pca_cols[2]],
        mode='markers',
        marker=dict(
            size=2,
            color=df[action_cols[0]],  # You can map color to any variable
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(
        title='Advanced 3D Scatter Plot: State PCA 1-3 Colored by Action Dimension 1',
        scene=dict(
            xaxis_title=state_pca_cols[0],
            yaxis_title=state_pca_cols[1],
            zaxis_title=state_pca_cols[2]
        )
    )

    fig.write_html(os.path.join(output_dir, 'advanced_3d_scatter.html'))


def main():
    log_dir = 'path_to_your_log_directory'  # Replace with your actual log directory
    output_dir = os.path.join(log_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # Load transitions data
    df = load_transitions(log_dir)

    # Initialize Dash app
    app = create_dash_app(df, output_dir)

    # Run the Dash app
    app.run_server(debug=True)
