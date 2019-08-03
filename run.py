import tensorflow as tf
import os
import cv2
import numpy as np
import math
import data_utils
import random
from skimage import io
import edsr
import mdsr_slim
from PIL import Image

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

class run:
    def __init__(self, config, ckpt_path, batch, epochs, B, F, lr, load_flag, meanBGR):
        self.config = config
        self.ckpt_path = ckpt_path
        self.batch = batch
        self.epochs = epochs
        self.B = B 
        self.F = F
        self.lr = lr
        self.load_flag = load_flag
        self.mean = meanBGR
    
    def train(self, imagefolder, validfolder):
        
        # Get all image paths from the training folder
        image_paths = data_utils.getpaths(imagefolder)
        
        # [TRAINING] Define 3 training datasets, one for each scale
        x2_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_dataset, 
                                                    output_types=(tf.float32, tf.float32), 
                                                    output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                    args=[image_paths, 2, self.mean])
        x2_dataset = x2_dataset.padded_batch(self.batch, padded_shapes=([None, None, 3],[None, None, 3]))
        x2_iterator = x2_dataset.make_initializable_iterator()
        
        x3_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_dataset, 
                                                    output_types=(tf.float32, tf.float32), 
                                                    output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                    args=[image_paths, 3, self.mean])
        x3_dataset = x3_dataset.padded_batch(self.batch, padded_shapes=([None, None, 3],[None, None, 3]))
        x3_iterator = x3_dataset.make_initializable_iterator()

        x4_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_dataset, 
                                                    output_types=(tf.float32, tf.float32), 
                                                    output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                    args=[image_paths, 4, self.mean])
        x4_dataset = x4_dataset.padded_batch(self.batch, padded_shapes=([None, None, 3],[None, None, 3]))
        x4_iterator = x4_dataset.make_initializable_iterator()

        # Get all image paths from the validation folder
        val_image_paths = data_utils.getpaths(validfolder)
        # [VALIDATION] Define 3 training datasets, one for each scale
        val_x2_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_val_dataset, 
                                                    output_types=(tf.float32, tf.float32), 
                                                    output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                    args=[val_image_paths, 2, self.mean])
        val_x2_dataset = val_x2_dataset.padded_batch(1, padded_shapes=([None, None, 3],[None, None, 3]))
        val_x2_iterator = val_x2_dataset.make_initializable_iterator()
        
        val_x3_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_val_dataset, 
                                                    output_types=(tf.float32, tf.float32), 
                                                    output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                    args=[val_image_paths, 3, self.mean])
        val_x3_dataset = val_x3_dataset.padded_batch(1, padded_shapes=([None, None, 3],[None, None, 3]))
        val_x3_iterator = val_x3_dataset.make_initializable_iterator()

        val_x4_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_val_dataset, 
                                                    output_types=(tf.float32, tf.float32), 
                                                    output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                    args=[val_image_paths, 4, self.mean])
        val_x4_dataset = val_x4_dataset.padded_batch(1, padded_shapes=([None, None, 3],[None, None, 3]))
        val_x4_iterator = val_x4_dataset.make_initializable_iterator()

        # A feedable iterator is defined by a handle placeholder and its structure.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, x2_dataset.output_types, x2_dataset.output_shapes)
        
        # Get next element from iterator
        LR, HR = iterator.get_next()
        
        print("Running MDSR.")
        mdsrObj = mdsr_slim.Mdsr(self.B, self.F)
        out, loss, train_op, psnr, lr = mdsrObj.model(x=LR, y=HR, lr=self.lr)

        # -- Training session
        with tf.Session(config=self.config) as sess:
            
            train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
            sess.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver()
            
            # Create check points directory if not existed, and load previous model if specified.
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            else:
                if os.path.isfile(self.ckpt_path + "mdsr_ckpt" + ".meta"):
                    if self.load_flag:
                        saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
                        print("\nLoaded checkpoint.")
                    if not self.load_flag:
                        print("No checkpoint loaded. Training from scratch.")
                else:
                    print("Previous checkpoint does not exists.")
            
            global_step = 0
            tf.convert_to_tensor(global_step)
            
            # The `Iterator.string_handle()` method returns a tensor that can be evaluated
            # and used to feed the `handle` placeholder.
            x2_iterator_handle = sess.run(x2_iterator.string_handle())
            x3_iterator_handle = sess.run(x3_iterator.string_handle())
            x4_iterator_handle = sess.run(x4_iterator.string_handle())

            sess.run(x2_iterator.initializer)
            sess.run(x3_iterator.initializer)
            sess.run(x4_iterator.initializer)

            val_x2_iterator_handle = sess.run(val_x2_iterator.string_handle())
            val_x3_iterator_handle = sess.run(val_x3_iterator.string_handle())
            val_x4_iterator_handle = sess.run(val_x4_iterator.string_handle())

            steps = 100000
            
            print("Training...")
            for e in range(1, self.epochs):
                
                sess.run(val_x2_iterator.initializer)
                sess.run(val_x3_iterator.initializer)
                sess.run(val_x4_iterator.initializer)

                train_loss, train_psnr = 0, 0
                
                for step in range(1,steps):
                    try:
                        # get batch of random scale
                        current_handle = None
                        r_scale = random.randint(2, 4)
                        if r_scale == 2:
                            current_handle = x2_iterator_handle
                        elif r_scale == 3:
                            current_handle = x3_iterator_handle
                        else:
                            current_handle = x4_iterator_handle
                        
                        o, l, t, l_rate = sess.run([out, loss, train_op, lr], feed_dict={handle:current_handle, 
                                                                                         mdsrObj.scale: r_scale,
                                                                                         mdsrObj.global_step: global_step})
                        train_loss += l
                        global_step += 1
                        
                        if step % 10000 == 0:
                            save_path = saver.save(sess, self.ckpt_path + "mdsr_ckpt")  
                            print("Step nr: [{}/{}] - Loss: {:.5f} - LR: {:5f}".format(step, steps, float(train_loss/step), l_rate))
                    
                    except tf.errors.OutOfRangeError:
                        #print("Iterator for scale {} has finished. Reinitializing...".format(r_scale))
                        if r_scale == 2:
                            sess.run(x2_iterator.initializer)
                        elif r_scale == 3:
                            sess.run(x3_iterator.initializer)
                        else:
                            sess.run(x4_iterator.initializer)
                        pass

                # Perform end-of-epoch calculations here.
                tot_psnr_x2, tot_psnr_x3, tot_psnr_x4 = 0, 0, 0
                im_counter = 0

                while True:
                    try:
                        p_x2 = sess.run([psnr], feed_dict={handle:val_x2_iterator_handle, 
                                                           mdsrObj.scale: 2})
                        p_x3 = sess.run([psnr], feed_dict={handle:val_x3_iterator_handle, 
                                                           mdsrObj.scale: 3})
                        p_x4 = sess.run([psnr], feed_dict={handle:val_x4_iterator_handle, 
                                                           mdsrObj.scale: 4})

                        tot_psnr_x2 += p_x2[0]                                                              
                        tot_psnr_x3 += p_x3[0]                                                              
                        tot_psnr_x4 += p_x4[0] 

                        im_counter += 1                                                             
                    except tf.errors.OutOfRangeError:
                        break
                
                print("Epoch nr: [{}/{}]  - Loss: {:.5f}".format(e, self.epochs, float(train_loss/steps))) 
                print("Valid set PSNR - [x2: {:.3f}] [x3: {:.3f}] [x4: {:.3f}]\n".format(tot_psnr_x2[0]/im_counter,
                                                                                 tot_psnr_x3[0]/im_counter,
                                                                                 tot_psnr_x4[0]/im_counter))
                
                save_path = saver.save(sess, self.ckpt_path + "mdsr_ckpt")   

            print("Training finished.")
            train_writer.close()

    def upscale(self, path, scale):
        """
        Upscales an image via model.
        """
        fullimg = cv2.imread(path, 3)

        floatimg = fullimg.astype(np.float32) - self.mean

        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

        with tf.Session(config=self.config) as sess:
            
            print("\nUpscale image by a factor of {}:\n".format(scale))
            
            # load the model
            ckpt_name = self.ckpt_path + "mdsr_ckpt" + ".meta"
            saver = tf.train.import_meta_graph(ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
            graph_def = sess.graph
            
            # get tensors
            LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
            HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")
            current_scale = graph_def.get_tensor_by_name("current_scale:0")

            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_, current_scale: scale})

            Y = output[0]
            HR_image = (Y + self.mean).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)

            bicubic_image = cv2.resize(fullimg, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            cv2.imshow('Original image', fullimg)
            cv2.imshow('MDSR HR image', HR_image)
            cv2.imshow('Bicubic HR image', bicubic_image)
            cv2.waitKey(0)

    def test(self, path, scale):
        """
        Test single image and calculate psnr.
        """
        fullimg = cv2.imread(path, 3)
        width = fullimg.shape[0]
        height = fullimg.shape[1]

        cropped = fullimg[0:(width - (width % scale)), 0:(height - (height % scale)), :]
        img = cv2.resize(cropped, None, fx=1. / scale, fy=1. / scale, interpolation=cv2.INTER_CUBIC)
        floatimg = img.astype(np.float32) - self.mean

        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

        with tf.Session(config=self.config) as sess:
            print("\nComparing MDSR with bicubic with scale {}:\n".format(scale))
            # load the model
            ckpt_name = self.ckpt_path + "mdsr_ckpt" + ".meta"
            saver = tf.train.import_meta_graph(ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
            graph_def = sess.graph
            
            LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
            HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")
            current_scale = graph_def.get_tensor_by_name("current_scale:0")
            
            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_, current_scale: scale})

            Y = output[0]
            HR_image = (Y + self.mean).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)

            bicubic_image = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            print("PSNR of  MDSR   upscaled image: {}".format(self.psnr(cropped, HR_image)))
            print("PSNR of bicubic upscaled image: {}".format(self.psnr(cropped, bicubic_image)))

            cv2.imshow('Original image', fullimg)
            cv2.imshow('MDSR HR image', HR_image)
            cv2.imshow('Bicubic HR image', bicubic_image)
            
            cv2.imwrite("./images/mdsrOutput.png", HR_image)
            cv2.imwrite("./images/bicubicOutput.png", bicubic_image)
            cv2.imwrite("./images/original.png", fullimg)
            cv2.imwrite("./images/input.png", img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def export(self, quant):
        print("Exporting model...")

        graph = tf.get_default_graph()
        with graph.as_default():
            with tf.Session(config=self.config) as sess:
                
                ### Restore checkpoint
                ckpt_name = self.ckpt_path + "mdsr_ckpt" + ".meta"
                saver = tf.train.import_meta_graph(ckpt_name)
                saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))

                # Return a serialized GraphDef representation of this graph
                graph_def = sess.graph.as_graph_def()

                # All variables to constants
                graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['NCHW_output'])
                
                # Optimize for inference
                graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, 
                                                                              ["IteratorGetNext"],
                                                                              ["NCHW_output"],
                                                                              tf.float32.as_datatype_enum)
                transforms = ["sort_by_execution_order"]
                pb_filename = "./models/MDSR.pb"
                if quant:
                    transforms = ["sort_by_execution_order", "quantize_weights"]
                    pb_filename = "./models/MDSRq.pb"

                graph_def = TransformGraph(graph_def, ["IteratorGetNext"], ["NCHW_output"], transforms)

                with tf.gfile.FastGFile(pb_filename, 'wb') as f:
                    f.write(graph_def.SerializeToString())

                tf.train.write_graph(graph_def, ".", 'train.pbtxt')

    def psnr(self, img1, img2):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))