import sys
import infer


def main():

    run_type = sys.argv[0]
    print(run_type)

    if run_type=='train':
        os.system('python train.py')

    elif run_type=='test':
        infer.infer(sys.argv[1])

    elif run_type=='vis_get':
    	infer.get_activations(sys.argv[1])

    else:
    	print("To run this script please enter either: 'train' or 'test <x>.png'")

if __name__ == '__main__':
  main()