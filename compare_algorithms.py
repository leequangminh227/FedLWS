"""
Script so sanh FedAvg va FedLWS
Chay ca 2 thuat toan va ve bieu do so sanh
"""

from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse


def run_algorithm(args, algorithm_name):
    """
    Chay mot thuat toan va tra ve ket qua
    
    Args:
        args: Arguments
        algorithm_name: 'fedavg' hoac 'fedlws'
    
    Returns:
        results: Dict chua test_acc, train_loss, time
    """
    print(f"\n{'='*70}")
    print(f"Running {algorithm_name.upper()}")
    print(f"{'='*70}\n")
    
    # Set server method
    args.server_method = algorithm_name
    
    # Set random seed
    setup_seed(args.random_seed)
    
    # Loading data
    data = Data(args)
    
    sample_size = []
    for i in range(args.node_num): 
        sample_size.append(len(data.train_loader[i]))
    size_weights = [i/sum(sample_size) for i in sample_size]
    central_node = Node(-1, data.test_loader[0], data.test_set, args)
    
    # Initialize client nodes
    client_nodes = {}
    for i in range(args.node_num): 
        client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args)
    
    # Results storage
    test_acc_recorder = []
    train_loss_recorder = []
    time_recorder = []
    
    # Start timing
    start_time = time.perf_counter()
    
    # Training loop
    for rounds in range(args.T):
        torch.cuda.empty_cache()
        round_start = time.perf_counter()
        
        print(f'\n{"="*60}', flush=True)
        print(f'{algorithm_name.upper()} - Round {rounds + 1}/{args.T}', flush=True)
        print(f'{"="*60}', flush=True)
        
        # LR scheduler
        lr_scheduler(rounds, client_nodes, args)
        current_lr = client_nodes[0].optimizer.param_groups[0]['lr']
        print(f'[INFO] Learning rate: {current_lr:.6f}', flush=True)
        
        # Client update
        print(f'[INFO] Starting client training...', flush=True)
        client_start = time.perf_counter()
        client_nodes, train_loss = Client_update(args, client_nodes, central_node)
        client_time = time.perf_counter() - client_start
        train_loss_recorder.append(train_loss)
        print(f'[INFO] Client training completed in {client_time:.2f}s', flush=True)
        
        # Client validation
        avg_client_acc = Client_validate(args, client_nodes)
        print(f'[INFO] Average client accuracy: {avg_client_acc:.2f}%', flush=True)
        
        # Partial client selection
        if args.select_ratio == 1.0:
            select_list = [idx for idx in range(len(client_nodes))]
        else:
            select_list = generate_selectlist(client_nodes, args.select_ratio)
        print(f'[INFO] Selected {len(select_list)} clients for aggregation', flush=True)
        
        # Server update
        print(f'[INFO] Server aggregating with {algorithm_name.upper()}...', flush=True)
        start = time.perf_counter()
        central_node = Server_update(args, central_node, client_nodes, select_list, size_weights)
        end = time.perf_counter()
        agg_time = end - start
        time_recorder.append(agg_time)
        print(f'[INFO] Aggregation completed in {agg_time:.4f}s', flush=True)
        
        # Test accuracy
        print(f'[INFO] Evaluating global model...', flush=True)
        acc = validate(args, central_node, which_dataset='local')
        test_acc_recorder.append(acc)
        
        round_time = time.perf_counter() - round_start
        
        # Summary
        print(f'\n[SUMMARY] Round {rounds+1}/{args.T}:', flush=True)
        print(f'  Test Accuracy:    {acc:.2f}%', flush=True)
        print(f'  Train Loss:       {train_loss:.4f}', flush=True)
        print(f'  Best Accuracy:    {max(test_acc_recorder):.2f}%', flush=True)
        print(f'  Round Time:       {round_time:.2f}s', flush=True)
        print(f'    - Client Training: {client_time:.2f}s', flush=True)
        print(f'    - Server Agg:      {agg_time:.4f}s', flush=True)
        
        # Progress bar
        progress = (rounds + 1) / args.T * 100
        bar_length = 40
        filled = int(bar_length * (rounds + 1) / args.T)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\nProgress: |{bar}| {progress:.1f}% ({rounds+1}/{args.T} rounds)', flush=True)
        
        # ETA
        if rounds > 0:
            avg_round_time = (time.perf_counter() - round_start) / (rounds + 1)
            remaining_rounds = args.T - (rounds + 1)
            eta_seconds = avg_round_time * remaining_rounds
            eta_minutes = eta_seconds / 60
            print(f'ETA: ~{eta_minutes:.1f} minutes', flush=True)
    
    # Final statistics
    best_acc = max(test_acc_recorder)
    avg_time = np.mean(time_recorder)
    final_acc = test_acc_recorder[-1]
    total_time = time.perf_counter() - start_time
    
    print(f"\n{'='*70}")
    print(f"{algorithm_name.upper()} - TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"\n[FINAL RESULTS]")
    print(f"  Final Accuracy:        {final_acc:.2f}%")
    print(f"  Best Accuracy:         {best_acc:.2f}%")
    print(f"  Avg Time per Round:    {avg_time:.4f}s")
    print(f"  Total Training Time:   {total_time/60:.2f} minutes")
    print(f"  Total Rounds:          {args.T}")
    print(f"{'='*70}\n")
    
    return {
        'test_acc': test_acc_recorder,
        'train_loss': train_loss_recorder,
        'time': time_recorder,
        'best_acc': best_acc,
        'final_acc': final_acc,
        'avg_time': avg_time
    }


def plot_comparison(results_fedavg, results_fedlws, args, save_dir='results'):
    """
    Ve bieu do so sanh giua FedAvg va FedLWS
    
    Args:
        results_fedavg: Ket qua cua FedAvg
        results_fedlws: Ket qua cua FedLWS
        args: Arguments
        save_dir: Thu muc luu ket qua
    """
    os.makedirs(save_dir, exist_ok=True)
    
    rounds = list(range(1, args.T + 1))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Test Accuracy
    ax1 = axes[0]
    ax1.plot(rounds, results_fedavg['test_acc'], 'b-o', label='FedAvg', linewidth=2, markersize=4)
    ax1.plot(rounds, results_fedlws['test_acc'], 'r-s', label='FedLWS', linewidth=2, markersize=4)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Train Loss
    ax2 = axes[1]
    ax2.plot(rounds, results_fedavg['train_loss'], 'b-o', label='FedAvg', linewidth=2, markersize=4)
    ax2.plot(rounds, results_fedlws['train_loss'], 'r-s', label='FedLWS', linewidth=2, markersize=4)
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Train Loss', fontsize=12)
    ax2.set_title('Train Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bar chart - Final comparison
    ax3 = axes[2]
    metrics = ['Final Acc\n(%)', 'Best Acc\n(%)', 'Avg Time\n(s)']
    fedavg_values = [
        results_fedavg['final_acc'],
        results_fedavg['best_acc'],
        results_fedavg['avg_time'] * 100  # Scale for visibility
    ]
    fedlws_values = [
        results_fedlws['final_acc'],
        results_fedlws['best_acc'],
        results_fedlws['avg_time'] * 100  # Scale for visibility
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, fedavg_values, width, label='FedAvg', color='skyblue')
    bars2 = ax3.bar(x + width/2, fedlws_values, width, label='FedLWS', color='salmon')
    
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.legend(fontsize=11)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'comparison_{args.dataset}_{args.local_model}_T{args.T}_nodes{args.node_num}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] Plot saved to: {filepath}")
    
    plt.show()
    plt.close()


def save_results_to_csv(results_fedavg, results_fedlws, args, save_dir='results'):
    """
    Luu ket qua ra file CSV
    
    Args:
        results_fedavg: Ket qua FedAvg
        results_fedlws: Ket qua FedLWS
        args: Arguments
        save_dir: Thu muc luu
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DataFrame
    rounds = list(range(1, args.T + 1))
    df = pd.DataFrame({
        'Round': rounds,
        'FedAvg_TestAcc': results_fedavg['test_acc'],
        'FedAvg_TrainLoss': results_fedavg['train_loss'],
        'FedAvg_Time': results_fedavg['time'],
        'FedLWS_TestAcc': results_fedlws['test_acc'],
        'FedLWS_TrainLoss': results_fedlws['train_loss'],
        'FedLWS_Time': results_fedlws['time']
    })
    
    # Save to CSV
    filename = f'comparison_{args.dataset}_{args.local_model}_T{args.T}_nodes{args.node_num}.csv'
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[SAVED] Results saved to: {filepath}")
    
    # Save summary
    summary = {
        'Algorithm': ['FedAvg', 'FedLWS'],
        'Final_Accuracy': [results_fedavg['final_acc'], results_fedlws['final_acc']],
        'Best_Accuracy': [results_fedavg['best_acc'], results_fedlws['best_acc']],
        'Avg_Time_per_Round': [results_fedavg['avg_time'], results_fedlws['avg_time']]
    }
    df_summary = pd.DataFrame(summary)
    
    summary_filename = f'summary_{args.dataset}_{args.local_model}_T{args.T}_nodes{args.node_num}.csv'
    summary_filepath = os.path.join(save_dir, summary_filename)
    df_summary.to_csv(summary_filepath, index=False)
    print(f"[SAVED] Summary saved to: {summary_filepath}")


def print_comparison_summary(results_fedavg, results_fedlws):
    """
    In ra ket qua so sanh
    """
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print("\n1. FINAL ACCURACY:")
    print(f"   FedAvg:  {results_fedavg['final_acc']:.2f}%")
    print(f"   FedLWS:  {results_fedlws['final_acc']:.2f}%")
    diff_final = results_fedlws['final_acc'] - results_fedavg['final_acc']
    print(f"   Difference: {diff_final:+.2f}% {'(FedLWS better)' if diff_final > 0 else '(FedAvg better)'}")
    
    print("\n2. BEST ACCURACY:")
    print(f"   FedAvg:  {results_fedavg['best_acc']:.2f}%")
    print(f"   FedLWS:  {results_fedlws['best_acc']:.2f}%")
    diff_best = results_fedlws['best_acc'] - results_fedavg['best_acc']
    print(f"   Difference: {diff_best:+.2f}% {'(FedLWS better)' if diff_best > 0 else '(FedAvg better)'}")
    
    print("\n3. AVERAGE TIME PER ROUND:")
    print(f"   FedAvg:  {results_fedavg['avg_time']:.4f}s")
    print(f"   FedLWS:  {results_fedlws['avg_time']:.4f}s")
    time_diff = results_fedlws['avg_time'] - results_fedavg['avg_time']
    print(f"   Difference: {time_diff:+.4f}s")
    
    print(f"\n{'='*70}")


def main():
    """
    Main function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compare FedAvg and FedLWS')
    
    # Data
    parser.add_argument('--noniid_type', type=str, default='dirichlet')
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--validate_batchsize', type=int, default=128)
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5)
    parser.add_argument('--dirichlet_alpha2', type=float, default=False)
    
    # System
    parser.add_argument('--device', type=str, default='0', help="GPU device")
    parser.add_argument('--node_num', type=int, default=10, help="Number of clients")
    parser.add_argument('--T', type=int, default=50, help="Number of rounds")
    parser.add_argument('--E', type=int, default=1, help="Local epochs")
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'fmnist', 'tinyimagenet'])
    parser.add_argument('--select_ratio', type=float, default=1.0)
    parser.add_argument('--local_model', type=str, default='ResNet20',
                       choices=['CNN', 'ResNet20', 'ResNet18', 'WRN56_4', 'Vit'])
    parser.add_argument('--random_seed', type=int, default=42)
    
    # FedLWS specific
    parser.add_argument('--min_tau', type=float, default=0.01)
    parser.add_argument('--max_tau', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.03)
    parser.add_argument('--longtail_clients', type=str, default="none")
    
    # Server
    parser.add_argument('--server_method', type=str, default='fedlws')
    parser.add_argument('--server_valid_ratio', type=float, default=0.02)
    parser.add_argument('--a', type=int, default=0.1)
    parser.add_argument('--b', type=int, default=0.1)
    
    # Client
    parser.add_argument('--client_method', type=str, default='local_train')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--client_valid_ratio', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.08)
    parser.add_argument('--local_wd_rate', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--mu', type=float, default=0.001)
    
    # Save directory
    parser.add_argument('--save_dir', type=str, default='comparison_results',
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    print(f"\n{'='*70}")
    print("🚀 FEDERATED LEARNING COMPARISON: FedAvg vs FedLWS")
    print(f"{'='*70}")
    print(f"\n📋 Configuration:")
    print(f"  Dataset:              {args.dataset}")
    print(f"  Model:                {args.local_model}")
    print(f"  Number of clients:    {args.node_num}")
    print(f"  Communication rounds: {args.T}")
    print(f"  Local epochs:         {args.E}")
    print(f"  Batch size:           {args.batchsize}")
    print(f"  Learning rate:        {args.lr}")
    print(f"  Non-IID (Dirichlet):  α={args.dirichlet_alpha}")
    print(f"  FedLWS beta:          {args.beta}")
    print(f"  Random seed:          {args.random_seed}")
    print(f"  Device:               {args.device}")
    print(f"  Save directory:       {args.save_dir}")
    print(f"{'='*70}\n")
    
    start_all = time.perf_counter()
    
    # Run FedAvg
    print(f"\n{'#'*70}")
    print(f"# PHASE 1/2: Running FedAvg")
    print(f"{'#'*70}\n")
    args_fedavg = deepcopy(args)
    results_fedavg = run_algorithm(args_fedavg, 'fedavg')
    
    # Run FedLWS
    print(f"\n{'#'*70}")
    print(f"# PHASE 2/2: Running FedLWS")
    print(f"{'#'*70}\n")
    args_fedlws = deepcopy(args)
    results_fedlws = run_algorithm(args_fedlws, 'fedlws')
    
    total_time = time.perf_counter() - start_all
    
    # Print comparison summary
    print_comparison_summary(results_fedavg, results_fedlws)
    
    # Save results to CSV
    save_results_to_csv(results_fedavg, results_fedlws, args, args.save_dir)
    
    # Plot comparison
    print(f"\n[INFO] Generating comparison plots...")
    plot_comparison(results_fedavg, results_fedlws, args, args.save_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ COMPARISON COMPLETED!")
    print(f"{'='*70}")
    print(f"\n📊 Results:")
    print(f"  Total Experiment Time: {total_time/60:.2f} minutes")
    print(f"  Results Directory:     {args.save_dir}/")
    print(f"\n📁 Generated Files:")
    filename_base = f'comparison_{args.dataset}_{args.local_model}_T{args.T}_nodes{args.node_num}'
    print(f"  - {filename_base}.png")
    print(f"  - {filename_base}.csv")
    print(f"  - summary_{args.dataset}_{args.local_model}_T{args.T}_nodes{args.node_num}.csv")
    print(f"\n{'='*70}\n")
    print(f"🎉 Done! Check results in: {args.save_dir}/\n")


if __name__ == '__main__':
    main()

