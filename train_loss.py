import matplotlib.pyplot as plt
import os
import sys
from tensorboard.backend.event_processing import event_accumulator
import argparse

def get_best_agent():
    """
    Find the best (most recently trained) agent in the logs directory.
    Returns the agent name with the highest number.
    """
    logs_dir = "./sawblade_logs"
    if not os.path.exists(logs_dir):
        return None
    
    agents = []
    for item in os.listdir(logs_dir):
        if os.path.isdir(f"{logs_dir}/{item}") and item.startswith('PPO_'):
            try:
                num = int(item.split('_')[1])
                agents.append((num, item))
            except (ValueError, IndexError):
                pass
    
    if not agents:
        return None
    
    # Return the agent with the highest number
    agents.sort(reverse=True)
    return agents[0][1]

def plot_ppo_losses(agent_name=None):
    """
    Plot training losses for a PPO agent from TensorBoard event files.
    
    Args:
        agent_name (str, optional): Name of the agent folder (e.g., 'PPO_1', 'PPO_2').
                                   If None, uses the best (most recent) agent.
    """
    # Auto-detect best agent if not specified
    if agent_name is None:
        agent_name = get_best_agent()
        if agent_name is None:
            print("Error: No agents found in ./sawblade_logs/")
            return
        print(f"Using best agent: {agent_name}")
    
    log_dir = f"./sawblade_logs/{agent_name}"
    
    # Check if directory exists
    if not os.path.exists(log_dir):
        print(f"Error: Directory {log_dir} does not exist!")
        print("Available agents:")
        if os.path.exists("./sawblade_logs"):
            for item in os.listdir("./sawblade_logs"):
                if os.path.isdir(f"./sawblade_logs/{item}"):
                    print(f"  - {item}")
        return
    
    # Create plots folder if it doesn't exist
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load event accumulator
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available scalars
    print(f"Available scalars in {agent_name}:")
    for scalar in ea.Tags()['scalars']:
        print(f"  - {scalar}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Metrics for {agent_name}', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('rollout/ep_len_mean', 'Episode Length (Mean)', axes[0, 0]),
        (   'rollout/ep_rew_mean', 'Episode Reward (Mean)', axes[0, 1]),
        ('train/policy_loss', 'Policy Loss', axes[1, 0]),
        ('train/value_loss', 'Value Loss', axes[1, 1]),
    ]
    
    for metric, title, ax in metrics_to_plot:
        try:
            events = ea.Scalars(metric)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            ax.plot(steps, values, linewidth=2, color='#1f77b4')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps', fontsize=10)
            ax.set_ylabel(title.split('(')[0].strip(), fontsize=10)
            ax.grid(True, alpha=0.3)
            
        except KeyError:
            ax.text(0.5, 0.5, f'{metric}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, color='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure with plot_{agentname} format in plots folder
    output_file = f"{plots_dir}/plot_{agent_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot PPO training losses from TensorBoard logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_loss.py              # Plots the best (most recent) agent
  python train_loss.py PPO_1        # Plots PPO_1
  python train_loss.py PPO_2        # Plots PPO_2
        """
    )
    parser.add_argument('agent_name', nargs='?', default=None,
                       help='Name of the agent to plot (e.g., PPO_1, PPO_2). If omitted, plots the best agent.')
    
    args = parser.parse_args()
    plot_ppo_losses(args.agent_name)
