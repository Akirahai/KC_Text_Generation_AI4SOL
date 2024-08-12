import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/Knowledge_Base/', help='Directory for data dir')
    # parser.add_argument('--seed', type=int, default=42, help='Seed to split data') #42
    # parser.add_argument('--seeds', type=int, nargs='+', default=[42, 50, 100], help='List of seeds to split data')
    parser.add_argument('--models', type=str, nargs='+', help='List of models to train')
    parser.add_argument('--num-classes', type=int, default=4, help='Num of grade')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.00009, help='Learning rate') #0.0001
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--model', type=str, help='Model name or path')
    parser.add_argument('--path', type=str, default= f"./result_second_ver") #Fix to your path to save model
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3], help='List of gpus to use')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--eval', type=str, default='test', help='Evaluation on test or valid set')
    parser.add_argument('--top-k', type=int, default=3, help='Top k accuracy')
    parser.add_argument('--experiment', type=str, default='1000_exp', help='Experiment name')
    parser.add_argument('--samples', type=int, default=100, help='Number of testing samples')
    
    return parser.parse_args()
    



# from utils import Math_Classification
# from utils import train
# from utils import validation



if __name__== "__main__":
    args = parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))
    
    import os
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    print(f"Using GPU: {GPU_list}")

    from libs import *
    args.best_metric = 0
    
    if args.use_gpu and torch.cuda.is_available(): 
        device = torch.device(f'cuda:1')  # Change to your suitable GPU device
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    #Login
    if args.model in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B-Instruct']:
        from huggingface_hub import login
        login()
    
    
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
        
        
        
    results = []
    current_time = args.experiment