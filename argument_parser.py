from argparse import ArgumentParser
import os
import sys
import json

def myParser():
    parser = ArgumentParser(description='Joint training of motion boundaries'
                                        'and occlusions (MONet).')
    parser.add_argument('--experiment_name', type=str, default='MONet_ft3d',
                        help='Name of the experiment.')
    parser.add_argument('--dataset_root', type=str, 
                        help='Path to the dataset for training or testing.')
    parser.add_argument('--flowEst_root', type=str, 
                        help='Path to the estimated flow maps corresponding to'
                             'data in dataset_root.')
    parser.add_argument('--gpu_device', default='0', help='GPU device ID.')
    parser.add_argument('--is_train', default=2, type=int,
                        help='Use train/valication/test split of the dataset:'
                             '0 - validation, 1 - training, 2 - testing.')
    parser.add_argument('--batch_size', default=1, type=int, 
                        help='Number of samples in a batch.')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Initial learning rate for training.')
    parser.add_argument('--load_weights', type=str,
                        help='Path to previously trained weights in the'
                             'experiment folder to load. If empty, dont load'
                             'any weights.')
    parser.add_argument('--init_epoch', default=0, type=int, 
                        help='Initial epoch number for training.')
    parser.add_argument('--num_epochs', default=100, type=int, 
                        help='Total number of epochs to train.')
    parser.add_argument('--sizeV', default=256, type=int,
                        help='Vertical size of the input to the network')
    parser.add_argument('--sizeH', default=448, type=int,
                        help='Horizontal size of the input to the network')
    parser.add_argument('--optimizer_type', default='adam', type=str, 
                        choices=['adam', 'sgd'],
                        help='Type of optimizer to use in training.')
    parser.add_argument('--save_preds', action='store_true', default=False, 
                        help='Save test predictions to prediction folder.')
    args = parser.parse_args()

    # save all arguments in a json file
    args_file = os.path.join('experiments/',args.experiment_name, 'args.json')
    if args.is_train!=1:
        if not os.path.isfile(args_file):
            raise IOError('`args.json` not found in {}'.format(args_file))
        
        print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)
        
        # check conflicts between loaded and given arguments.
        for key, value in args.__dict__.items():
            if key in args_resumed:
                resumed_value = args_resumed[key]
                if resumed_value != value and key != 'is_train' and \
                   key != 'batch_size' and key != 'gpu_device' and \
                   key != 'load_weights' and key != 'dataset_root' and \
                   key != 'flowEst_root' :
                    print('Warning: For the argument `{}` we are using the'
                          ' loaded value `{}`. The provided value was `{}`'
                          '.'.format(key, resumed_value, value))
                    args.__dict__[key] = resumed_value
            else:
                print('Warning: A new argument was added since the last run:'
                      ' `{}`. Using the new value: `{}`.'.format(key, value))
    elif args.is_train==1:
        if os.path.exists('experiments/'+args.experiment_name):
            if os.listdir('experiments/'+args.experiment_name) and \
               args.init_epoch!=0:
                print('The directory {} already exists. Resuming training at'
                      ' init_epoch = {} with the new input parameters.'.format(
                      'experiments/'+args.experiment_name, args.init_epoch))
        else:
            os.makedirs('experiments/'+args.experiment_name)

        # save the passed arguments 
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, 
                      sort_keys=True)

    # Print parameters
    if args.is_train==1:
        print('Training using the following parameters:')
    else:
        print('Testing using the following parameters:')
    for key, value in sorted(vars(args).items()):
        print('{}: {}'.format(key, value))

    return args
