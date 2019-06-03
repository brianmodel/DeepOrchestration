# Using Cuda Docker Container

Once you create your deep learning model and are ready to train, follow these steps to ensure that you are using the GPUs on the server.

1. In the terminal type

   ```sh
   pydocker
   ```

   The following text should appear: "Must add python file to run as parameter!" This means that the script is on the path and should be working.

2. In your python main.py file (or whatever the entry point of your file) write the following code to make sure that tensorflow can see the gpus. 

   ```python
   from tensorflow.python.client import device_lib
   print(device_lib.list_local_devices())
   ```

   When running 

   ```sh
   pydocker YOU_FILE_NAME.py
   ```

   you should see GPU as a device type in the terminal output. Make sure you are running the file with pydocker and **not** python, otherwise you will not be using the GPUs

3. Once you ensure that tensorflow can see the GPUs, you can start training. Again, **use the pydocker command** to run the script to have the program run on the GPUs instead of the CPUs.

— Brian Model