import h5py

# Open the HDF5 file
#file_path = 'output_TW.h5'  # Replace with your file path
#file_path = 'samples/TbarBQ_t-channel.h5'  # Replace with your file path
#file_path='example_signal.h5'
file_path='TTto2L2Nu.h5'

with h5py.File(file_path, 'r') as f:
    # List all groups and datasets
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")

    # Recursively print the file structure
    f.visititems(print_attrs)

    # If you know the specific dataset you want to inspect, you can access it like this:
    #dataset_name = 'sys_weights'  # Replace with the actual dataset name
    #dataset_name ='event_info'
    dataset_name ='norm_weights'
    #dataset_name ='sys_weights'
 
   #dataset_name='jec_msoftdrop_up'
    #dataset_name='jet1_extraInfo'
    #dataset_name='preselection_eff'
    if dataset_name in f:
        data = f[dataset_name][:]
        print(f"\nData in {dataset_name}:")
        print(data)

        # >>> CHANGED PART <<<: Print the number of events
        print(f"Number of events: {data.shape[0]}")  
    else:
        print(f"\n{dataset_name} not found in the file.")

