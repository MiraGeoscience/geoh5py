GIFtools Groups
===============

GIFtools groups are containers used to organize UBC-GIF inversions and their
parameter forms.

GIFtools Project
----------------

**UUID : {585b3218-c24b-41fe-ad1f-24d5e6e8348a}**

Top-level container for GIFtools inversion groups.

Attributes
^^^^^^^^^^

:giftoolsVersion: ``int`` (default ``1``)
    Version number stored on the project group.
:Can add group: ``int``, 0 or (default) 1
    Controls whether child groups can be added from the user interface.
:gif_parameters: list of ``dict``
    Per-child parameter forms managed by the project group.


GIF Executables
---------------

**UUID : {afae95ef-c2a7-4aec-9800-0d19bd2c2c07}**

*Not yet geoh5py implemented*

*To be documented*

Each implemented executable group stores a ``parameters`` dictionary that
contains the inversion form (options, linked entities, and execution values).

All executable groups below derive from
``geoh5py.groups.giftools.base.BaseGIFtoolsGroup``.

gzinv3d
^^^^^^^

**UUID : {20eb4ff8-bdfe-43f3-8745-f418dcc9e14a}**

**Version : 6**

Inversion group for UBC-GZINV3D gravity inversion.

gzfor3d
^^^^^^^

**UUID : {a4857df0-d175-4824-ac5d-cecfdcc2f20b}**

*Not yet geoh5py implemented*

*To be documented*

magfor3d
^^^^^^^^

**UUID : {6b8189ac-a479-4fe7-b4fc-92279aee5a41}**

*Not yet geoh5py implemented*

*To be documented*

maginv3d
^^^^^^^^

**UUID : {b99e8db8-e118-4042-864e-9e1128f2d1e6}**

**Version : 6**

Inversion group for UBC-MAGINV3D scalar magnetic inversion.

mvifwd
^^^^^^

**UUID : {14c41f47-bcee-4a63-8192-fa42a1741052}**

*Not yet geoh5py implemented*

*To be documented*

ggfor3d
^^^^^^^

**UUID : {c8a8424d-ab12-482e-82ee-b198fcfd5859}**

*Not yet geoh5py implemented*

*To be documented*

gginv3d
^^^^^^^

**UUID : {0f080369-b3a3-464c-83fa-9b3c1efa9895}**

**Version : 6**

Inversion group for UBC-GGINV3D gravity gradiometry inversion.

mviinv
^^^^^^

**UUID : {9472b5cb-a285-4257-a2e8-68a3d33aa1f2}**

**Version : 3**

Inversion group for UBC-MVIINV magnetic vector inversion.

octgrvde
^^^^^^^^

**UUID : {4e043415-a0ea-4cef-bf89-2771e27b346c}**

**Version : 1**

Inversion group for UBC-OCTGRVDE octree gravity inversion.

octmagde
^^^^^^^^

**UUID : {f8217512-296d-4cc0-afcb-6c07a20581fe}**

*Not yet geoh5py implemented*

*To be documented*

dcinv3d
^^^^^^^

**UUID : {ae416ab8-0e72-4f37-8873-5cc0909433bb}**

*Not yet geoh5py implemented*

*To be documented*

ipinv3d
^^^^^^^

**UUID : {9f9543a0-e857-4a56-ab66-9f21e2b002c6}**

**Version : 5**

Inversion group for UBC-IPINV3D induced polarization inversion.

e3dmt
^^^^^

**UUID : {8cf239e3-63a6-4813-adf8-9714293b602e}**

*Not yet geoh5py implemented*

*To be documented*

dcoctree_inv
^^^^^^^^^^^^

**UUID : {54d296de-0588-472c-9a62-480098303394}**

**Version : 20200508**

Inversion group for UBC-DCOctree direct current octree inversion.

dcoctree_fwd
^^^^^^^^^^^^

**UUID : {a522d641-6cb7-421b-836b-a14c0d9c7801}**

*Not yet geoh5py implemented*

*To be documented*

ipoctree_inv
^^^^^^^^^^^^

**UUID : {d9fd455e-ea94-40f5-9d86-e7c49c7b5005}**

**Version : 20200508**

Inversion group for UBC-IPOctree induced polarization octree inversion.

dcipf3d
^^^^^^^

**UUID : {59b5338d-596c-4049-9aa4-6979700e00ff}**

*Not yet geoh5py implemented*

*To be documented*

e3d
^^^

**UUID : {9a0b9d39-9e6d-409e-a7cd-ffc72474feed}**

**Version : 1**

Inversion group for UBC-E3D electromagnetic inversion.

H3DTDinv
^^^^^^^^

**UUID : {4f864000-15a1-4381-afec-b274ab765568}**

**Version : 1**

Inversion group for UBC-H3DTDInv time-domain electromagnetic inversion.
