	�o'��@�o'��@!�o'��@	$"b'-f�?$"b'-f�?!$"b'-f�?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:�o'��@��.���?Aa����@Y�V��,s�?rEagerKernelExecute 0*	��S�8�'A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorLl>��I�@!��
$��X@)Ll>��I�@1��
$��X@:Preprocessing2F
Iterator::Model���)x�?!z�$�?)��Tka�?1�8�@#��?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismI+�����?!r�?�$c�?))��q��?1N5�}YF�?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch�~�n�?!&�2[�p?)�~�n�?1&�2[�p?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapx_��I�@!?_����X@)�p��[um?1{�7z�K>?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9$"b'-f�?Iw'�t��X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��.���?��.���?!��.���?      ��!       "      ��!       *      ��!       2	a����@a����@!a����@:      ��!       B      ��!       J	�V��,s�?�V��,s�?!�V��,s�?R      ��!       Z	�V��,s�?�V��,s�?!�V��,s�?b      ��!       JCPU_ONLYY$"b'-f�?b qw'�t��X@