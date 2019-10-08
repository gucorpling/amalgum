import tensorflow as tf
import numpy as np
import codecs, io


infile = "C:\\tmp\\seg_data\\wiki.he.60k.justnums.vec"
infile = "C:\\Uni\\RST\\edusegmenter\\vec\\cc.zh.300.vec"


with open(infile,encoding="utf8") as f:
	lines = (" ".join(line.split()[1:]) for line in f if not len(line.strip())==0)
	FH = np.loadtxt(lines, delimiter=' ', skiprows=1)
#filecp = io.open(infile, encoding = 'utf8')
embedding = np.loadtxt(infile,skiprows=1)

embedding_dim = embedding.shape[1]
vocab_size = embedding.shape[0]


W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=True, name="W")




#W = tf.get_variable(name="W", shape=embedding.shape, initializer=tf.constant_initializer(embedding), trainable=True)




embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

# ...

with tf.Session() as sess:

	sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
	embedding_saver = tf.train.Saver({"W": W})
	embedding_saver.save(sess,"C:\\Uni\\RST\\edusegmenter\\vec\\cc.zh.300.bin")

#W = tf.get_variable(name="W", shape=embedding.shape, initializer=tf.constant_initializer(embedding), trainable=False)


#sess = tf.Session()
#embedding_saver.restore(sess, "checkpoint_filename.ckpt")