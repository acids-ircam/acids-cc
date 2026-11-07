``cached_conv``
===============

.. currentmodule:: cached_conv

Mode
----

.. autofunction:: use_cached_conv
.. autofunction:: is_cached

Testing helpers
---------------

.. autofunction:: chunk_process
.. autofunction:: test_equal

Factories
---------

Each factory forwards its arguments to the eager or the cached class.

.. list-table::
   :header-rows: 1

   * - Factory
     - Eager
     - Cached
   * - ``Conv1d``
     - :class:`~cached_conv.conv_1d.Conv1d`
     - :class:`~cached_conv.conv_1d.CachedConv1d`
   * - ``ConvTranspose1d``
     - :class:`~cached_conv.conv_1d.ConvTranspose1d`
     - :class:`~cached_conv.conv_1d.CachedConvTranspose1d`
   * - ``AlignBranches1d``
     - :class:`~cached_conv.utils.Branches`
     - :class:`~cached_conv.conv_1d.AlignBranches1d`
   * - ``Conv2d``
     - :class:`~cached_conv.conv_2d.Conv2d`
     - :class:`~cached_conv.conv_2d.CachedConv2d`
   * - ``ConvTranspose2d``
     - :class:`~cached_conv.conv_2d.ConvTranspose2d`
     - :class:`~cached_conv.conv_2d.CachedConvTranspose2d`
   * - ``AlignBranches2d``
     - :class:`~cached_conv.utils.Branches`
     - :class:`~cached_conv.conv_2d.AlignBranches2d`
   * - ``Conv3d``
     - :class:`~cached_conv.conv_3d.Conv3d`
     - :class:`~cached_conv.conv_3d.CachedConv3d`
   * - ``ConvTranspose3d``
     - :class:`~cached_conv.conv_3d.ConvTranspose3d`
     - :class:`~cached_conv.conv_3d.CachedConvTranspose3d`
   * - ``AlignBranches3d``
     - :class:`~cached_conv.utils.Branches`
     - :class:`~cached_conv.conv_2d.AlignBranches2d`
   * - ``WindowedConv1d``
     - :class:`~cached_conv.variants.WindowedConv1d`
     - :class:`~cached_conv.variants.CachedWindowedConv1d`
   * - ``WindowedConv2d``
     - :class:`~cached_conv.variants.WindowedConv2d`
     - :class:`~cached_conv.variants.CachedWindowedConv2d`
   * - ``WindowedConv3d``
     - :class:`~cached_conv.variants.WindowedConv3d`
     - :class:`~cached_conv.variants.CachedWindowedConv3d`
   * - ``ModulatedConv1d``
     - :class:`~cached_conv.variants.ModulatedConv1d`
     - :class:`~cached_conv.variants.CachedModulatedConv1d`
   * - ``ModulatedConv2d``
     - :class:`~cached_conv.variants.ModulatedConv2d`
     - :class:`~cached_conv.variants.CachedModulatedConv2d`
   * - ``ModulatedConv3d``
     - :class:`~cached_conv.variants.ModulatedConv3d`
     - :class:`~cached_conv.variants.CachedModulatedConv3d`
   * - ``Upsample1d``
     - :class:`~cached_conv.upsample.Upsample1d`
     - :class:`~cached_conv.upsample.CachedUpsample1d`
   * - ``Upsample2d``
     - :class:`~cached_conv.upsample.Upsample2d`
     - :class:`~cached_conv.upsample.CachedUpsample2d`
   * - ``Upsample3d``
     - :class:`~cached_conv.upsample.Upsample3d`
     - :class:`~cached_conv.upsample.CachedUpsample3d`
