import pickle

with open(r'..\data\models\logs\my_model_final_2\model_0009.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Access the state_dict of the loaded model
state_dict = model_data['model_state']

# Iterate through the state_dict and extract weight and bias values for each layer
for name, param in state_dict.items():
    if name.endswith('.weight'):
        print(f'Layer: {name}')
        print('Weight:', param)

        # Find the corresponding bias parameter
        bias_name = name.replace('.weight', '.bias')
        if bias_name in state_dict:
            bias_param = state_dict[bias_name]
            print('Bias:', bias_param)
        else:
            print('No Bias for this layer')
        print('-' * 40)
