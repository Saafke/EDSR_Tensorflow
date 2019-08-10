import tensorflow as tf
import os
import cv2
import numpy as np
import math
import data_utils
from skimage import io
import edsr
from PIL import Image

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

class run:
    def __init__(self, config, ckpt_path, scale, batch, epochs, B, F, lr, load_flag, meanBGR):
        self.config = config
        self.ckpt_path = ckpt_path
        self.scale = scale
        self.batch = batch
        self.epochs = epochs
        self.B = B
        self.F = F
        self.lr = lr
        self.load_flag = load_flag
        self.mean = meanBGR

    def train(self, imagefolder, validfolder):

        # Create training dataset
        train_image_paths = data_utils.getpaths(imagefolder)
        train_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_dataset,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                 args=[train_image_paths, self.scale, self.mean])
        train_dataset = train_dataset.padded_batch(self.batch, padded_shapes=([None, None, 3],[None, None, 3]))

        # Create validation dataset
        val_image_paths = data_utils.getpaths(validfolder)
        val_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_val_dataset,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])),
                                                 args=[val_image_paths, self.scale, self.mean])
        val_dataset = val_dataset.padded_batch(1, padded_shapes=([None, None, 3],[None, None, 3]))

        # Make the iterator and its initializers
        train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = train_val_iterator.make_initializer(train_dataset)
        val_initializer = train_val_iterator.make_initializer(val_dataset)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        LR, HR = iterator.get_next()

        # Edsr model
        print("\nRunning EDSR.")
        edsrObj = edsr.Edsr(self.B, self.F, self.scale)
        out, loss, train_op, psnr, ssim, lr = edsrObj.model(x=LR, y=HR, lr=self.lr)

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
                # else:
                #     if os.path.isfile("./CKPT_dir/x2/" + "edsr_ckpt" + ".meta"):
                #         saver.restore(sess, tf.train.latest_checkpoint("./CKPT_dir/x2/"))
                #         print("Previous checkpoint does not exists. Will load model from x2")
                #     else:
                #         print("No checkpoint loaded. Training from scratch.")

            global_step = 0
            tf.convert_to_tensor(global_step)

            train_val_handle = sess.run(train_val_iterator.string_handle())

            print("Training...")
            for e in range(1, self.epochs+1):

                sess.run(train_initializer)
                step, train_loss = 0, 0

                try:
                    while True:
                        o, l, t, l_rate = sess.run([out, loss, train_op, lr], feed_dict={handle:train_val_handle,
                                                                                         edsrObj.global_step: global_step})
                        train_loss += l
                        step += 1
                        global_step += 1

                        if step % 1000 == 0:
                            save_path = saver.save(sess, self.ckpt_path + "edsr_ckpt")
                            print("Step nr: [{}/{}] - Loss: {:.5f} - Lr: {:.7f}".format(step, "?", float(train_loss/step), l_rate))

                except tf.errors.OutOfRangeError:
                    pass

                # Perform end-of-epoch calculations here.
                sess.run(val_initializer)
                tot_val_psnr, tot_val_ssim, val_im_cntr = 0, 0, 0
                try:
                    while True:
                        val_psnr, val_ssim = sess.run([psnr, ssim], feed_dict={handle:train_val_handle})

                        tot_val_psnr += val_psnr[0]
                        tot_val_ssim += val_ssim[0]
                        val_im_cntr += 1

                except tf.errors.OutOfRangeError:
                    pass

                print("Epoch nr: [{}/{}]  - Loss: {:.5f} - val PSNR: {:.3f} - val SSIM: {:.3f}\n".format(e,
                                                                                                         self.epochs,
                                                                                                         float(train_loss/step),
                                                                                                         (tot_val_psnr / val_im_cntr),
                                                                                                         (tot_val_ssim / val_im_cntr)))
                save_path = saver.save(sess, self.ckpt_path + "edsr_ckpt")

            print("Training finished.")
            train_writer.close()

    def upscale(self, path):
        """
        Upscales an image via model. This loads a checkpoint, not a .pb file.
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
            cv2.imshow('EDSR upscaled image', HR_image)
            cv2.imshow('Bicubic upscaled image', bicubic_image)
            cv2.waitKey(0)

        sess.close()

    def test(self, path):
        """
        Test single image and calculate psnr. This loads a checkpoint, not a .pb file.
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

            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})

            Y = output[0]
            HR_image = (Y + self.mean).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)

            bicubic_image = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

            print(np.amax(Y), np.amax(LR_input_))
            print("PSNR of  EDSR   upscaled image: {}".format(self.psnr(cropped, HR_image)))
            print("PSNR of bicubic upscaled image: {}".format(self.psnr(cropped, bicubic_image)))

            cv2.imshow('Original image', fullimg)
            cv2.imshow('EDSR upscaled image', HR_image)
            cv2.imshow('Bicubic upscaled image', bicubic_image)

            cv2.imwrite("./images/EdsrOutput.png", HR_image)
            cv2.imwrite("./images/BicubicOutput.png", bicubic_image)
            cv2.imwrite("./images/original.png", fullimg)
            cv2.imwrite("./images/input.png", img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        sess.close()

    def load_pb(self, path_to_pb):
        with tf.gfile.GFile(path_to_pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph

    def testFromPb(self, path):
        """
        Test single image and calculate psnr. This loads a .pb file.
        """
        # Read model
        pbPath = "./models/EDSR_x{}.pb".format(self.scale)

        # Get graph
        graph = self.load_pb(pbPath)

        fullimg = cv2.imread(path, 3)
        width = fullimg.shape[0]
        height = fullimg.shape[1]

        cropped = fullimg[0:(width - (width % self.scale)), 0:(height - (height % self.scale)), :]
        img = cv2.resize(cropped, None, fx=1. / self.scale, fy=1. / self.scale, interpolation=cv2.INTER_CUBIC)
        floatimg = img.astype(np.float32) - self.mean

        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

        LR_tensor = graph.get_tensor_by_name("IteratorGetNext:0")
        HR_tensor = graph.get_tensor_by_name("NHWC_output:0")

        with tf.Session(graph=graph) as sess:
            print("Loading pb...")
            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})
            Y = output[0]
            HR_image = (Y + self.mean).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)

            bicubic_image = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

            print(np.amax(Y), np.amax(LR_input_))
            print("PSNR of  EDSR   upscaled image: {}".format(self.psnr(cropped, HR_image)))
            print("PSNR of bicubic upscaled image: {}".format(self.psnr(cropped, bicubic_image)))

            cv2.imshow('Original image', fullimg)
            cv2.imshow('EDSR upscaled image', HR_image)
            cv2.imshow('Bicubic upscaled image', bicubic_image)

            cv2.imwrite("./images/EdsrOutput.png", HR_image)
            cv2.imwrite("./images/BicubicOutput.png", bicubic_image)
            cv2.imwrite("./images/original.png", fullimg)
            cv2.imwrite("./images/input.png", img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Done.")

        sess.close()

    def upscaleFromPb(self, path):
        """
        Upscale single image by desired model. This loads a .pb file.
        """
        # Read model
        pbPath = "./models/EDSR_x{}.pb".format(self.scale)

        # Get graph
        graph = self.load_pb(pbPath)

        fullimg = cv2.imread(path, 3)
        floatimg = fullimg.astype(np.float32) - self.mean
        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)

        LR_tensor = graph.get_tensor_by_name("IteratorGetNext:0")
        HR_tensor = graph.get_tensor_by_name("NHWC_output:0")

        with tf.Session(graph=graph) as sess:
            print("Loading pb...")
            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})
            Y = output[0]
            HR_image = (Y + self.mean).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)

            bicubic_image = cv2.resize(fullimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

            cv2.imshow('Original image', fullimg)
            cv2.imshow('EDSR upscaled image', HR_image)
            cv2.imshow('Bicubic upscaled image', bicubic_image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        sess.close()

    def export(self, quant):
        print("Exporting model...")

        export_dir = "./models/"
        if not os.path.exists(export_dir):
                os.makedirs(export_dir)

        export_file = "EDSRorig_x{}.pb".format(self.scale)

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
                
                # Implement certain file shrinking transforms. 2 is recommended.
                transforms = ["sort_by_execution_order"]
                if quant == 1:
                    print("Rounding weights for export.")
                    transforms = ["sort_by_execution_order", "round_weights"]
                    export_file = "EDSR_x{}_q1.pb".format(self.scale)
                if quant == 2:
                    print("Quantizing for export.")
                    transforms = ["sort_by_execution_order", "quantize_weights"]
                    export_file = "EDSR_x{}.pb".format(self.scale)
                if quant == 3:
                    print("Round weights and quantizing for export.")
                    transforms = ["sort_by_execution_order", "round_weights", "quantize_weights"]
                    export_file = "EDSR_x{}_q3.pb".format(self.scale)

                graph_def = TransformGraph(graph_def, ["IteratorGetNext"],
                                                      ["NCHW_output"],
                                                      transforms)
                
                print("Exported file = {}".format(export_dir+export_file))
                with tf.gfile.GFile(export_dir + export_file, 'wb') as f:
                    f.write(graph_def.SerializeToString())

                tf.train.write_graph(graph_def, ".", 'train.pbtxt')

        sess.close()

    def psnr(self, img1, img2):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))