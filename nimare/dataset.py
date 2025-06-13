"""Classes for representing datasets of images and/or coordinates."""

import copy
import inspect
import json
import logging
import os.path as op
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from nilearn._utils import load_niimg

from nimare.base import NiMAREBase
from nimare.utils import (
    _dict_to_coordinates,
    _dict_to_df,
    _listify,
    _transform_coordinates_to_space,
    _try_prepend,
    _validate_df,
    _validate_images_df,
    get_masker,
    get_template,
    mm2vox,
)

LGR = logging.getLogger(__name__)


class Dataset(NiMAREBase):
    """Storage container for a coordinate- and/or image-based meta-analytic dataset/database.

    .. versionchanged:: 0.0.9

        * [ENH] Add merge method to Dataset class

    .. versionchanged:: 0.0.8

        * [FIX] Set ``nimare.dataset.Dataset.basepath`` in :func:`update_path` using absolute path.

    Parameters
    ----------
    source : :obj:`str` or :obj:`dict`
        JSON file containing dictionary with database information or the dict()
        object

    target : :obj:`str`, optional
        Desired coordinate space for coordinates. Names follow NIDM convention.
        Default is 'mni152_2mm' (MNI space with 2x2x2 voxels).
        This parameter has no impact on images.

    mask : :obj:`str`, :class:`~nibabel.nifti1.Nifti1Image`, \
    :class:`~nilearn.input_data.NiftiMasker` or similar, or None, optional
        Mask(er) to use. If None, uses the target space image, with all
        non-zero voxels included in the mask.

    Attributes
    ----------
    space : :obj:`str`
        Standard space. Same as ``target`` parameter.

    Notes
    -----
    Images loaded into a Dataset are assumed to be in the same space.
    If images have different resolutions or affines from the Dataset's masker,
    then they will be resampled automatically, at the point where they're used,
    by :obj:`Dataset.masker`.
    """

    _id_cols = ["id", "study_id", "contrast_id"]

    def __init__(self, source, target="mni152_2mm", mask=None):
        if isinstance(source, str):
            with open(source, "r") as f_obj:
                data = json.load(f_obj)
        elif isinstance(source, dict):
            data = source
        else:
            raise Exception("`source` needs to be a file path or a dictionary")

        # Datasets are organized by study, then experiment
        # To generate unique IDs, we combine study ID with experiment ID
        # 1. build list of ids
        id_columns = ["id", "study_id", "contrast_id"]
        all_ids = [
            [f"{pid}-{expid}", pid, expid]
            for pid in data.keys()
            for expid in data[pid]["contrasts"].keys()
        ]
        # Create IDs DataFrame and set initial IDs
        id_df = pd.DataFrame(columns=id_columns, data=all_ids)
        id_df = id_df.set_index("id", drop=False)
        # Set IDs directly to avoid recursion
        self._id_array = np.sort(id_df.index.values)

        # 2. Set up Masker
        if mask is None:
            mask = get_template(target, mask="brain")
        self._masker = get_masker(mask)  # Store directly
        self.space = target

        # 3. Create DataFrames without triggering sorts in setters
        self._annotations = _dict_to_df(id_df, data, key="labels")
        self._coordinates = _dict_to_coordinates(data, masker=self._masker, space=self.space)
        self._images = _validate_images_df(_dict_to_df(id_df, data, key="images"))
        self._metadata = _dict_to_df(id_df, data, key="metadata")
        self._texts = _dict_to_df(id_df, data, key="text")
        self.basepath = None

        # 4. Sort all DataFrames once
        for df_name in ("_annotations", "_coordinates", "_images", "_metadata", "_texts"):
            df = getattr(self, df_name)
            if df is not None and not df.empty:
                df.sort_values(by="id", inplace=True)

        # Handle z-statistics at initialization
        if (hasattr(self, "_coordinates")
                and "z_stat" in self._coordinates.columns
                and not self._coordinates["z_stat"].isna().any()):
            z_stats = self._coordinates["z_stat"].astype(float)
            self._coordinates["z_stat"] = z_stats
            pos_stats = (z_stats >= 0).any()
            neg_stats = (z_stats < 0).any()
            if pos_stats and neg_stats:
                warnings.warn(
                    "Coordinates dataset contains both positive and negative z_stats. "
                    "The algorithms currently implemented in NiMARE are designed for "
                    "one-sided tests. This might lead to unexpected results."
                )

    def __repr__(self):
        """Show basic Dataset representation.

        It's basically the same as the NiMAREBase representation, but with the number of
        experiments in the Dataset represented as well.
        """
        # Get default parameter values for the object
        signature = inspect.signature(self.__init__)
        defaults = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        # Eliminate any sub-parameters (e.g., parameters for a Estimator's KernelTransformer),
        # as well as default values
        params = self.get_params()
        params = {k: v for k, v in params.items() if "__" not in k}
        # Parameter "target" is stored as attribute "space"
        # and we want to show it regardless of whether it's the default or not
        params["space"] = self.space
        params.pop("target")
        params = {k: v for k, v in params.items() if defaults.get(k) != v}

        # Convert to strings
        param_strs = []
        for k, v in params.items():
            if isinstance(v, str):
                # Wrap string values in single quotes
                param_str = f"{k}='{v}'"
            else:
                # Keep everything else as-is based on its own repr
                param_str = f"{k}={v}"
            param_strs.append(param_str)

        params_str = ", ".join(param_strs)
        params_str = f"{len(self.ids)} experiments{', ' if params_str else ''}{params_str}"
        rep = f"{self.__class__.__name__}({params_str})"
        return rep

    @property
    def ids(self):
        """numpy.ndarray: 1D array of identifiers in Dataset.

        The associated setter for this property is private, as ``Dataset.ids`` is immutable.
        """
        return self._id_array

    def _set_ids(self, ids):
        """Store IDs array directly in sorted order.
        
        Parameters
        ----------
        ids : array-like
            Array to store as the Dataset's IDs.
        """
        sorted_ids = np.sort(np.asarray(ids))
        assert isinstance(sorted_ids, np.ndarray) and sorted_ids.ndim == 1
        self._id_array = sorted_ids

    @property
    def masker(self):
        """:class:`nilearn.input_data.NiftiMasker` or similar: Masker object.

        Defines the space and location of the area of interest (e.g., 'brain').
        """
        return self._masker

    @masker.setter
    def masker(self, mask):
        mask = get_masker(mask)
        if hasattr(self, "_masker") and not np.array_equal(
            self._masker.mask_img.affine, mask.mask_img.affine
        ):
            # This message does not have an associated effect,
            # since matrix indices are calculated as necessary
            LGR.warning("New masker does not match old masker. Space is assumed to be the same.")
        self._masker = mask

    @property
    def annotations(self):
        """:class:`pandas.DataFrame`: Labels describing studies in the dataset.

        Each study/experiment has its own row.
        Columns correspond to individual labels (e.g., 'emotion'), and may
        be prefixed with a feature group including two underscores
        (e.g., 'Neurosynth_TFIDF__emotion').
        """
        return self._annotations

    @annotations.setter
    def annotations(self, df):
        _validate_df(df)
        self._annotations = df.sort_values(by="id")

    @property
    def coordinates(self):
        """:class:`pandas.DataFrame`: Coordinates in the dataset.

        .. versionchanged:: 0.0.10

            The coordinates attribute no longer includes the associated matrix indices
            (columns 'i', 'j', and 'k'). These columns are calculated as needed.

        Each study has one row for each peak.
        Columns include ['x', 'y', 'z'] (peak locations in mm) and 'space' (Dataset's space).
        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, df):
        _validate_df(df)
        self._coordinates = df.sort_values(by="id")

    @property
    def images(self):
        """:class:`pandas.DataFrame`: Images in the dataset.

        Each image type has its own column (e.g., 'z') with absolute paths to
        files and each study has its own row.
        Additionally, relative paths to image files are stored in columns with
        the suffix '__relative' (e.g., 'z__relative').

        Warnings
        --------
        Images are assumed to be in the same space, although they may have
        different resolutions and affines. Images will be resampled as needed
        at the point where they are used, via :obj:`Dataset.masker`.
        """
        return self._images

    @images.setter
    def images(self, df):
        _validate_df(df)
        self._images = _validate_images_df(df).sort_values(by="id")

    @property
    def metadata(self):
        """:class:`pandas.DataFrame`: Metadata describing studies in the dataset.

        Each metadata field has its own column (e.g., 'sample_sizes') and each study
        has its own row.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, df):
        _validate_df(df)
        self._metadata = df.sort_values(by="id")

    @property
    def texts(self):
        """:class:`pandas.DataFrame`: Texts in the dataset.

        Each text type has its own column (e.g., 'abstract') and each study
        has its own row.
        """
        return self._texts

    @texts.setter
    def texts(self, df):
        _validate_df(df)
        self._texts = df.sort_values(by="id")

    def slice(self, ids):
        """Create a new dataset with only requested IDs.

        Parameters
        ----------
        ids : array_like
            List of study IDs to include in new dataset

        Returns
        -------
        new_dset : :obj:`~nimare.dataset.Dataset`
            Reduced Dataset containing only requested studies.
            
        Notes
        -----
        This optimized version uses copy instead of deepcopy where possible,
        and avoids DataFrame copies when not needed.
        """
        # Create shallow copy and set IDs directly
        new_dset = copy.copy(self)
        new_dset._id_array = np.sort(np.asarray(ids))

        # Filter DataFrames using views where possible
        for attribute in ("annotations", "coordinates", "images", "metadata", "texts"):
            df = getattr(self, f"_{attribute}")
            if df is not None and not df.empty:
                # Create filtered view of the DataFrame
                filtered_df = df.loc[df["id"].isin(ids)].copy()  # Copy only the filtered data
                setattr(new_dset, f"_{attribute}", filtered_df)

        return new_dset

    def merge(self, right):
        """Merge two Datasets.

        .. versionadded:: 0.0.9

        Parameters
        ----------
        right : :obj:`~nimare.dataset.Dataset`
            Dataset to merge with.

        Returns
        -------
        :obj:`~nimare.dataset.Dataset`
            A Dataset of the two merged Datasets.
        """
        assert isinstance(right, Dataset)
        shared_ids = np.intersect1d(self.ids, right.ids)
        if shared_ids.size:
            raise Exception("Duplicate IDs detected in both datasets.")

        all_ids = np.concatenate((self.ids, right.ids))
        new_dset = copy.deepcopy(self)
        new_dset._id_array = np.sort(all_ids)

        # Merge DataFrames efficiently
        for attribute in ("annotations", "coordinates", "images", "metadata", "texts"):
            df1 = getattr(self, f"_{attribute}")
            df2 = getattr(right, f"_{attribute}")
            if df1 is not None and df2 is not None:
                # Use concat with copy=False for better performance
                new_df = pd.concat([df1, df2], ignore_index=True, copy=False)
                new_df.sort_values(by="id", inplace=True)
                new_df.reset_index(drop=True, inplace=True)
                new_df = new_df.where(~new_df.isna(), None)
                setattr(new_dset, f"_{attribute}", new_df)

        new_dset.coordinates = _transform_coordinates_to_space(
            new_dset.coordinates,
            self.masker,
            self.space,
        )

        return new_dset

    def update_path(self, new_path):
        """Update paths to images.

        Prepends new path to the relative path for files in Dataset.images.

        Parameters
        ----------
        new_path : :obj:`str`
            Path to prepend to relative paths of files in Dataset.images.
        """
        self.basepath = op.abspath(new_path)
        df = self.images
        relative_path_cols = [c for c in df if c.endswith("__relative")]
        for col in relative_path_cols:
            abs_col = col.replace("__relative", "")
            if abs_col in df.columns:
                LGR.info(f"Overwriting images column {abs_col}")
            df[abs_col] = df[col].apply(_try_prepend, prefix=self.basepath)
        self.images = df

    def copy(self):
        """Create a copy of the Dataset."""
        return copy.deepcopy(self)

    def get(self, dict_, drop_invalid=True):
        """Retrieve files and/or metadata from the current Dataset.

        Parameters
        ----------
        dict_ : :obj:`dict`
            Dictionary specifying images or metadata to collect.
            Keys should be variables to be used as keys for results dictionary.
            Values should be tuples with two values:
            type (e.g., 'image' or 'metadata') and specific field corresponding
            to column of type-specific DataFrame (e.g., 'z' or 'sample_sizes').
        drop_invalid : :obj:`bool`, optional
            Whether to automatically ignore any studies without the required data or not.
            Default is False.

        Returns
        -------
        results : :obj:`dict`
            A dictionary of lists of requested data. Keys correspond to the keys in ``dict_``.

        Examples
        --------
        >>> dset.get({'z_maps': ('image', 'z'), 'sample_sizes': ('metadata', 'sample_sizes')})
        >>> dset.get({'coordinates': ('coordinates', None)})
        """
        results = {}
        results["id"] = self.ids
        keep_idx = np.arange(len(self.ids), dtype=int)
        for k, vals in dict_.items():
            if vals[0] == "image":
                temp = self.get_images(imtype=vals[1])
            elif vals[0] == "metadata":
                temp = self.get_metadata(field=vals[1])
            elif vals[0] == "coordinates":
                dset_coord_groupby_id = dict(iter(self.coordinates.groupby("id")))
                temp = [
                    dset_coord_groupby_id[id_] if id_ in dset_coord_groupby_id.keys() else None
                    for id_ in self.ids
                ]
            elif vals[0] == "annotations":
                dset_annot_groupby_id = dict(iter(self.annotations.groupby("id")))
                temp = [
                    dset_annot_groupby_id[id_] if id_ in dset_annot_groupby_id.keys() else None
                    for id_ in self.ids
                ]
            else:
                raise ValueError(f"Input '{vals[0]}' not understood.")

            results[k] = temp
            temp_keep_idx = np.where([t is not None for t in temp])[0]
            keep_idx = np.intersect1d(keep_idx, temp_keep_idx)

        # reduce
        if drop_invalid and (len(keep_idx) != len(self.ids)):
            LGR.info(f"Retaining {len(keep_idx)}/{len(self.ids)} studies")
        elif len(keep_idx) != len(self.ids):
            raise Exception(
                f"Only {len(keep_idx)}/{len(self.ids)} in Dataset contain the necessary data. "
                "If you want to analyze the subset of studies with required data, "
                "set `drop_invalid` to True."
            )

        for k in results:
            results[k] = [results[k][i] for i in keep_idx]
            if dict_.get(k, [None])[0] in ("coordinates", "annotations"):
                results[k] = pd.concat(results[k])

        return results

    @lru_cache(maxsize=128)
    def _generic_column_getter(self, attr, ids=None, column=None, ignore_columns=None):
        """Get data from DataFrame attributes with caching for better performance.
        
        Uses lru_cache decorator to cache results for repeated access patterns.

        Parameters
        ----------
        attr : str
            Name of attribute to access (_annotations, _coordinates, etc)
        ids : tuple or str or None
            Study IDs to retrieve. Must be tuple for caching.
        column : str or None
            Column name to extract from DataFrame
        ignore_columns : tuple or None
            Columns to ignore in results. Must be tuple for caching.

        Returns
        -------
        list or str
            Requested data or first item if ids is str and column given
        """
        # Convert lists to tuples for caching
        if ignore_columns is not None:
            ignore_columns = tuple(ignore_columns)
        if ids is not None and not isinstance(ids, str):
            ids = tuple(ids)

        # Set up ignore columns
        if ignore_columns is None:
            ignore_columns = tuple(self._id_cols)
        else:
            ignore_columns = tuple(list(self._id_cols) + list(ignore_columns))
        
        # Get DataFrame and check if we need first item
        df = getattr(self, f"_{attr}")
        return_first = isinstance(ids, str) and column is not None
        id_list = [ids] if isinstance(ids, str) else ids

        # Process column access
        if column is not None:
            # Validate column exists
            if column not in df.columns:
                available = [c for c in df.columns if c not in ignore_columns]
                raise ValueError(
                    f"{column} not found in {attr}. Available columns: {', '.join(available)}"
                )
            
            # Get data for specified column
            if id_list is not None:
                result = df.loc[df["id"].isin(id_list), column].tolist()
            else:
                result = df[column].tolist()

        # Get available columns
        else:
            available_cols = [c for c in df.columns if c not in ignore_columns]
            if id_list is not None:
                subset = df[df["id"].isin(id_list)]
                result = [c for c in available_cols if subset[c].any()]
            else:
                result = [c for c in available_cols if df[c].any()]

        # Return first item if needed
        return result[0] if return_first and result else result

    def get_labels(self, ids=None):
        """Extract list of labels for which studies in Dataset have annotations.

        Parameters
        ----------
        ids : :obj:`list`, optional
            A list of IDs in the Dataset for which to find labels. Default is
            None, in which case all labels are returned.

        Returns
        -------
        labels : :obj:`list`
            List of labels for which there are annotations in the Dataset.
        """
        if not isinstance(ids, list) and ids is not None:
            ids = _listify(ids)

        result = [c for c in self.annotations.columns if c not in self._id_cols]
        if ids is not None:
            temp_annotations = self.annotations.loc[self.annotations["id"].isin(ids)]
            res = temp_annotations[result].any(axis=0)
            result = res.loc[res].index.tolist()

        return result

    def get_texts(self, ids=None, text_type=None):
        """Extract list of texts of a given type for selected IDs.

        Parameters
        ----------
        ids : :obj:`list`, optional
            A list of IDs in the Dataset for which to find texts. Default is
            None, in which case all texts of requested type are returned.
        text_type : :obj:`str`, optional
            Type of text to extract. Corresponds to column name in
            Dataset.texts DataFrame. Default is None.

        Returns
        -------
        texts : :obj:`list`
            List of texts of requested type for selected IDs.
        """
        # Convert ids to tuple for caching
        if ids is not None and not isinstance(ids, str):
            ids = tuple(ids)
        result = self._generic_column_getter("texts", ids=ids, column=text_type)
        return result

    def get_metadata(self, ids=None, field=None):
        """Get metadata from Dataset.

        Parameters
        ----------
        ids : :obj:`list`, optional
            A list of IDs in the Dataset for which to find metadata. Default is
            None, in which case all metadata of requested type are returned.
        field : :obj:`str`, optional
            Metadata field to extract. Corresponds to column name in
            Dataset.metadata DataFrame. Default is None.

        Returns
        -------
        metadata : :obj:`list`
            List of values of requested type for selected IDs.
        """
        # Convert ids to tuple for caching
        if ids is not None and not isinstance(ids, str):
            ids = tuple(ids)
        result = self._generic_column_getter("metadata", ids=ids, column=field)
        return result

    def get_images(self, ids=None, imtype=None):
        """Get images of a certain type for a subset of studies in the dataset.

        Parameters
        ----------
        ids : :obj:`list`, optional
            A list of IDs in the Dataset for which to find images. Default is
            None, in which case all images of requested type are returned.
        imtype : :obj:`str`, optional
            Type of image to extract. Corresponds to column name in
            Dataset.images DataFrame. Default is None.

        Returns
        -------
        images : :obj:`list`
            List of images of requested type for selected IDs.
        """
        # Convert inputs to hashable types for caching
        if ids is not None and not isinstance(ids, str):
            ids = tuple(ids)
        ignore_columns = tuple(["space"] + [c for c in self.images.columns if c.endswith("__relative")])
        result = self._generic_column_getter(
            "images",
            ids=ids,
            column=imtype,
            ignore_columns=ignore_columns
        )
        return result

    def get_studies_by_label(self, labels=None, label_threshold=0.001):
        """Extract list of studies with a given label.

        .. versionchanged:: 0.0.10

            Fix bug in which all IDs were returned when a label wasn't present in the Dataset.

        .. versionchanged:: 0.0.9

            Default value for label_threshold changed to 0.001.

        Parameters
        ----------
        labels : :obj:`list`, optional
            List of labels to use to search Dataset. If a contrast has all of
            the labels above the threshold, it will be returned.
            Default is None.
        label_threshold : :obj:`float`, optional
            Default is 0.5.

        Returns
        -------
        found_ids : :obj:`list`
            A list of IDs from the Dataset found by the search criteria.
        """
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            raise ValueError(f"Argument 'labels' cannot be {type(labels)}")

        missing_labels = [label for label in labels if label not in self.annotations.columns]
        if missing_labels:
            raise ValueError(f"Missing label(s): {', '.join(missing_labels)}")

        temp_annotations = self.annotations[self._id_cols + labels]
        found_rows = (temp_annotations[labels] >= label_threshold).all(axis=1)
        if any(found_rows):
            found_ids = temp_annotations.loc[found_rows, "id"].tolist()
        else:
            found_ids = []

        return found_ids

    def get_studies_by_mask(self, mask):
        """Extract list of studies with at least one coordinate in mask.

        Parameters
        ----------
        mask : img_like
            Mask across which to search for coordinates.

        Returns
        -------
        found_ids : :obj:`list`
            A list of IDs from the Dataset with at least one focus in the mask.
        """
        mask = load_niimg(mask)
        dset_mask = self.masker.mask_img

        if not np.array_equal(dset_mask.affine, mask.affine):
            LGR.warning("Mask affine does not match Dataset affine. Assuming same space.")

        # Get cached coordinates and IDs
        coords, coord_ids = self._coordinate_array
        
        # Convert to voxel coordinates once
        dset_ijk = mm2vox(coords, mask.affine)
        
        # Get mask data efficiently
        mask_data = mask.get_fdata()
        
        # Create boolean mask for valid coordinates
        valid_coords = (
            (dset_ijk >= 0) &
            (dset_ijk < np.array(mask_data.shape)[:, None])
        ).all(axis=0)

        # Index into mask using valid coordinates
        in_mask = np.zeros(len(dset_ijk), dtype=bool)
        if valid_coords.any():
            valid_ijk = dset_ijk[:, valid_coords]
            in_mask[valid_coords] = mask_data[
                valid_ijk[0],
                valid_ijk[1],
                valid_ijk[2]
            ].astype(bool)

        # Get unique IDs using numpy operations
        found_ids = list(np.unique(coord_ids[in_mask]))
        return found_ids

    # Cache coordinate array to avoid repeated DataFrame operations
    @property
    def _coordinate_array(self):
        """Cache the coordinate array for faster distance calculations."""
        if not hasattr(self, '_cached_coordinate_array'):
            self._cached_coordinate_array = self.coordinates[["x", "y", "z"]].values
            self._cached_coordinate_ids = self.coordinates["id"].values
        return self._cached_coordinate_array, self._cached_coordinate_ids

    def get_studies_by_coordinate(self, xyz, r=20):
        """Extract list of studies with at least one focus within radius of requested coordinates.

        Parameters
        ----------
        xyz : (X x 3) array_like
            List of coordinates against which to find studies.
        r : :obj:`float`, optional
            Radius (in mm) within which to find studies. Default is 20mm.

        Returns
        -------
        found_ids : :obj:`list`
            A list of IDs from the Dataset with at least one focus within
            radius r of requested coordinates.
        """
        from scipy.spatial.distance import cdist

        xyz = np.asarray(xyz)
        assert xyz.shape[1] == 3 and xyz.ndim == 2
        
        # Get cached coordinate array and IDs
        coords, coord_ids = self._coordinate_array
        
        # Compute distances using vectorized operation
        distances = cdist(xyz, coords)
        matches = coord_ids[np.any(distances <= r, axis=0)]
        
        # Get unique IDs using numpy operations instead of pandas
        found_ids = list(np.unique(matches))
        return found_ids
