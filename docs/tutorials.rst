Tutorials
============

This page contains a few tutorials explaining the usage of the main functions in TIME.

Fixel weight maps
-----------------

Fixel weights maps can be obtained using the :meth:`TIME.core.get_fixel_weight` function::

	import nibabel as nib

	t0 = nib.load("peak_file_0.nii.gz").get_fdata()
	t1 = nib.load("peak_file_1.nii.gz").get_fdata()

	trk = load_tractogram("trk_file.trk", 'same')

	fixel_weights,_,_ = get_fixel_weight(trk, [t0, t1])

Example
----------------

A complete example code of the main function is available in the :class:`TIME.example` file.

.. literalinclude:: ../TIME/example.py
    :linenos:
    :language: python
    :lines: 11-

.. note::  More tutorials will be added in the near future.