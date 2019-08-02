import tensorflow as tf
import os
import cv2
import numpy as np
import math
import data_utils
from skimage import io
import edsr
import mdsr_slim
from PIL import Image

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

class run:
    def __init__(self, config, ckpt_path, scale, mdsrFlag, batch, epochs, B, F, lr, load_flag, meanBGR):
        self.config = config
        self.ckpt_path = ckpt_path
        self.scale = scale
        self.mdsrFlag = mdsrFlag
        self.batch = batch
        self.epochs = epochs
        self.B = B 
        self.F = F
        self.lr = lr
        self.load_flag = load_flag
        self.mean = meanBGR
    
    def train(self, imagefolder):
        
        # Create training dataset iterator
        image_paths = data_utils.getpaths(imagefolder)
        #x2
        x2_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_dataset, 
                                                 output_types=(tf.float32, tf.float32), 
                                                 output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                 args=[image_paths, 2, self.mean])
        x2_dataset = x2_dataset.padded_batch(self.batch, padded_shapes=([None, None, 3],[None, None, 3]))
        
        #x3
        x3_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_dataset, 
                                                 output_types=(tf.float32, tf.float32), 
                                                 output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                 args=[image_paths, 3, self.mean])
        x3_dataset = x3_dataset.padded_batch(self.batch, padded_shapes=([None, None, 3],[None, None, 3]))

        #x4
        x4_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_dataset, 
                                                 output_types=(tf.float32, tf.float32), 
                                                 output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                 args=[image_paths, 4, self.mean])
        x4_dataset = x4_dataset.padded_batch(self.batch, padded_shapes=([None, None, 3],[None, None, 3]))
        

        train_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        
        x2_initializer = train_iterator.make_initializer(x2_dataset)
        x3_initializer = train_iterator.make_initializer(x3_dataset)
        x4_initializer = train_iterator.make_initializer(x4_dataset)
        
        iter = dataset.make_initializable_iterator()
        LR, HR = iter.get_next()
        
        if self.mdsrFlag:
            print("Running MDSR.")
            out, loss, train_op, psnr = mdsr_slim.model(x=LR, y=HR, B=16, F=64, scale=self.scale, batch=self.batch, lr=self.lr)
        else: # EDSR
            print("Running EDSR.")
            out, loss, train_op, psnr = edsr.model(x=LR, y=HR, B=self.B, F=self.F, scale=self.scale, batch=self.batch, lr=self.lr)
        
        # -- Training session
        with tf.Session(config=self.config) as sess:
            
            train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
            sess.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver()
            
            # Create check points directory if not existed, and load previous model if specified.
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            else:
                if os.path.isfile(self.ckpt_path + "edsr_ckpt" + ".meta"):
                    if self.load_flag:
                        saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
                        print("\nLoaded checkpoint.")
                    if not self.load_flag:
                        print("No checkpoint loaded. Training from scratch.")
                else:
                    print("Previous checkpoint does not exists.")

            print("Training...")
            for e in range(1, self.epochs):
                sess.run(iter.initializer)
                step, train_loss, train_psnr = 0, 0, 0 
    
                while True:
                    try:
                        o, l, t, ps = sess.run([out, loss, train_op, psnr])
                        train_loss += l
                        train_psnr += (np.mean(np.asarray(ps)))
                        step += 1
                        
                        if step % 1000 == 0:
                            save_path = saver.save(sess, self.ckpt_path + "edsr_ckpt")  
                            print("Step nr: [{}/{}] - Loss: {:.5f}".format(step, "?", float(train_loss/step)))
                    
                    except tf.errors.OutOfRangeError:
                        break

                # Perform end-of-epoch calculations here.
                print("Epoch nr: [{}/{}]  - Loss: {:.5f} - Valid set PSNR: {:.3f}\n".format(e,
                                                                                            self.epochs,
                                                                                            float(train_loss/step),
                                                                                            self.validTest()))
                save_path = saver.save(sess, self.ckpt_path + "edsr_ckpt")   

            print("Training finished.")
            train_writer.close()

    def validTest(self):
        """
        Tests model on a validation set.
        """
        inputs = list()
        tot_psnr = 0
        im_cnt = 0
        imageFolder = "/home/weber/Documents/gsoc/datasets/Set14"
        im_paths = data_utils.getpaths(imageFolder)

        with tf.Session(config=self.config) as sessx:
            
            ckpt_name = self.ckpt_path + "edsr_ckpt" + ".meta"
            saverx = tf.train.import_meta_graph(ckpt_name)
            saverx.restore(sessx, tf.train.latest_checkpoint(self.ckpt_path))
            graph_def = sessx.graph
            LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
            HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")

            # prepare image, upscale and calc psnr
            for p in im_paths:

                fullimg = cv2.imread(p, 3)
                width = fullimg.shape[0]
                height = fullimg.shape[1]

                cropped = fullimg[0:(width - (width % self.scale)), 0:(height - (height % self.scale)), :]
                img = cv2.resize(cropped, None, fx=1. / self.scale, fy=1. / self.scale, interpolation=cv2.INTER_CUBIC)
                floatimg = img.astype(np.float32) - self.mean

                inp = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)
                
                output = sessx.run(HR_tensor, feed_dict={LR_tensor: inp})

                Y = output[0]
                HR_image = (Y + self.mean).clip(min=0, max=255)
                HR_image = (HR_image).astype(np.uint8)

                tot_psnr += self.psnr(cropped, HR_image)
                im_cnt += 1
        
        return tot_psnr / im_cnt

    def upscale(self, path):
        """
        Upscales an image via model.
        """
        fullimg = cv2.imread(path, 3)

        floatimg = fullimg.astype(np.float32) - self.mean

        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

        with tf.Session(config=self.config) as sess:
            print("\nUpscale image by a factor of {}:\n".format(self.scale))
            # load the model
            ckpt_name = self.ckpt_path + "edsr_ckpt" + ".meta"
            saver = tf.train.import_meta_graph(ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
            graph_def = sess.graph
            LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
            HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")

            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})

            Y = output[0]
            HR_image = (Y + self.mean).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)

            bicubic_image = cv2.resize(fullimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

            cv2.imshow('Original image', fullimg)
            cv2.imshow('HR image', HR_image)
            cv2.imshow('Bicubic HR image', bicubic_image)
            cv2.waitKey(0)

    def test(self, path):
        """
        Test single image and calculate psnr.
        """
        fullimg = cv2.imread(path, 3)
        width = fullimg.shape[0]
        height = fullimg.shape[1]

        cropped = fullimg[0:(width - (width % self.scale)), 0:(height - (height % self.scale)), :]
        img = cv2.resize(cropped, None, fx=1. / self.scale, fy=1. / self.scale, interpolation=cv2.INTER_CUBIC)
        floatimg = img.astype(np.float32) - self.mean

        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

        with tf.Session(config=self.config) as sess:
            print("\nTest model with psnr:\n")
            # load the model
            ckpt_name = self.ckpt_path + "edsr_ckpt" + ".meta"
            saver = tf.train.import_meta_graph(ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
            graph_def = sess.graph
            LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
            HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")
            print("Loaded model.")
            
            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})

            Y = output[0]
            HR_image = (Y + self.mean).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)

            bicubic_image = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

            print(np.amax(Y), np.amax(LR_input_))
            print("PSNR of  EDSR   upscaled image: {}".format(self.psnr(cropped, HR_image)))
            print("PSNR of bicubic upscaled image: {}".format(self.psnr(cropped, bicubic_image)))

            cv2.imshow('Original image', fullimg)
            cv2.imshow('HR image', HR_image)
            cv2.imshow('Bicubic HR image', bicubic_image)
            
            cv2.imwrite("./images/edsrOutput.png", HR_image)
            cv2.imwrite("./images/bicubicOutput.png", bicubic_image)
            cv2.imwrite("./images/original.png", fullimg)
            cv2.imwrite("./images/input.png", img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def export(self):
        print("Exporting model...")

        graph = tf.get_default_graph()
        with graph.as_default():
            with tf.Session(config=self.config) as sess:
                
                ### Restore checkpoint
                ckpt_name = self.ckpt_path + "edsr_ckpt" + ".meta"
                saver = tf.train.import_meta_graph(ckpt_name)
                saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))

                # Return a serialized GraphDef representation of this graph
                graph_def = sess.graph.as_graph_def()

                # All variables to constants
                graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['NCHW_output'])
                
                # Optimize for inference
                graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ["IteratorGetNext"],
                                                                            ["NCHW_output"],  # ["NHWC_output"],
                                                                            tf.float32.as_datatype_enum)

                graph_def = TransformGraph(graph_def, ["IteratorGetNext"], ["NCHW_output"], ["sort_by_execution_order"])

                with tf.gfile.FastGFile('./models/EDSR_x{}.pb'.format(self.scale), 'wb') as f:
                    f.write(graph_def.SerializeToString())

                tf.train.write_graph(graph_def, ".", 'train.pbtxt')

    def psnr(self, img1, img2):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))