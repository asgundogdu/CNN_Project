import sys, os




def main():

    run_type = sys.argv[1]
    print('*'*120)
    print('*'*120)
    print('*'*120)
    print(run_type)

    if run_type=='train':
    	import train
        # os.system('python3 train.py')
        train.run_train()

    elif run_type=='test':
    	import infer
        infer.infer(sys.argv[2])
        infer.get_activations(sys.argv[2])

    else:
    	print("To run this script please enter either: 'train' or 'test <x>.png'")

if __name__ == '__main__':
  main()