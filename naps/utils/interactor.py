#!/usr/bin/env python
import os

import numpy as np
from shapely import area
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import unary_union


class Interactor:
    def __init__(
        self,
        name="",
        frame=-1,
        nodes=[],
        coords_array=np.array([]),
        skeleton={},
        model={},
        interactions={},
        **kwargs,
    ):

        # Assign the name and frame of the interactor
        self.name = name
        self.frame = frame

        # Assign the nodes, require to assign the coords_array
        self._nodes = nodes

        # Create a dicts to store the skeleton, models, and interactions
        self._skeleton = {}
        self._models = {}
        self._interactions = interactions

        # Update the skeleton using the coords_array and model
        for _sk in skeleton:
            self._updateSkeleton(coords_array, **_sk)

        # Update the skeleton using the interactions using the skeleton
        for _m in model:
            self._updateModel(**_m)

    def _updateSkeleton(
        self, coords_array, name="", shape="", nodes=[], buffer=0, exclude_nodes=[]
    ):

        # Check if shape has all nodes
        if exclude_nodes:
            nodes = [_n for _n in self._nodes if _n not in exclude_nodes]

        # Confirm a name was given for the skeleton
        if not name:
            raise Exception(
                f"No name given for skeleton with shape ({shape}) and nodes ({', '.join(nodes)})"
            )

        # Check the shape is allowed
        if shape not in ["line", "reduced-line", "point", "hull"]:
            raise Exception(f"The shape ({shape}) is not supported")

        # Confirm the nodes are supported
        if shape in ["line", "reduced-line"] and len(nodes) <= 1:
            raise Exception(
                f"The shape ({shape}) requires two or more nodes, found {len(nodes)} nodes."
            )
        elif shape == "hull" and len(nodes) <= 1:
            raise Exception(
                f"The shape ({shape}) requires two or more nodes, found {len(nodes)} nodes."
            )
        elif shape == "point" and len(nodes) != 1:
            raise Exception(
                f"The shape ({shape}) requires a single node, found {len(nodes)} nodes."
            )

        # Assign values
        node_coords = [coords_array[self._nodes.index(_n)] for _n in nodes]

        # Update the skeleton w/ the correct shape and values
        if shape == "line":
            self._skeleton[name] = LineString(node_coords).buffer(buffer)
        if shape == "reduced-line":
            line_string = LineString(node_coords[:2])
            node_coords[0] = line_string.interpolate(buffer)
            if len(node_coords) == 2:
                node_coords[-1] = line_string.interpolate(line_string.length - buffer)
                self._skeleton[name] = LineString(node_coords).buffer(buffer)
            else:
                end_line_string = LineString(node_coords[-2:])
                node_coords[-1] = end_line_string.interpolate(
                    end_line_string.length - buffer
                )
                self._skeleton[name] = LineString(node_coords).buffer(buffer)
        elif shape == "hull":
            self._skeleton[name] = MultiPoint(node_coords).convex_hull.buffer(buffer)
        elif shape == "point":
            self._skeleton[name] = Point(node_coords[0]).buffer(buffer)

    def _updateModel(self, name="", includes=[], excludes=[]):

        # Confirm a name was given for the interaction
        if not name:
            raise Exception(
                f"No name given for interaction model with includes ({', '.join(includes)}) and excludes ({', '.join(excludes)})"
            )

        # Create the interaction with the inclusion skeleton nodes
        interaction = unary_union([self._skeleton[_ic] for _ic in includes])

        # Check if there are exclusion skeleton nodes
        if len(excludes):

            # Create the interaction to exclude
            exclude_interaction = unary_union([self._skeleton[_ic] for _ic in excludes])

            # Create the interaction without the exclusion
            interaction = interaction.difference(exclude_interaction)

        # Update the interaction
        self._models[name] = interaction

    def interacts(self, other_interactor):
        def _processInteraction(
            name="", origin="", dest="", symmetical=False, return_int=False
        ):
            def _interact(origin_interactor, dest_interactor):

                # Check if two interact
                interact = origin_interactor._models[origin].intersects(
                    dest_interactor._models[dest]
                )
                return interact

            def _reportInteract(interact, origin_interactor, dest_interactor):

                # Return int if specified
                if return_int:
                    interact = int(interact)

                # Return the interaction
                return {
                    "Origin interactor": origin_interactor.name,
                    "Destination interactor": dest_interactor.name,
                    "Interaction Name": name,
                    "Interaction Frame": self.frame,
                    "Origin Area": area(origin_interactor._models[origin]),
                    "Destination Area": area(dest_interactor._models[dest])
                }

            # Confirm a name was given for the interaction
            if not name:
                raise Exception(
                    f"No name given for interaction with origin ({origin}) and dest ({dest})"
                )

            # Confirm a origin and dest were given
            if not origin:
                raise Exception(f"No origin given for interaction ({name})")
            if not dest:
                raise Exception(f"No dest given for interaction ({name})")

            # Confirm the origin and dest are in models
            if origin not in self._models:
                raise Exception(
                    f"Given origin ({origin}) not found among interaction models"
                )
            if dest not in self._models:
                raise Exception(
                    f"Given dest ({dest}) not found among interaction models"
                )

            # Confirm the type of symmetical
            if not isinstance(symmetical, bool):
                raise Exception(f"symmetical must be boolean")

            # Report symmetical results
            if symmetical:
                interaction_bool = _interact(self, other_interactor)
                if not interaction_bool:
                    return []
                interaction = _reportInteract(interaction_bool, self, other_interactor)
                return [interaction]

            # Report nonsymmetical results
            return_interactions = []

            # Test for an interaction
            interaction_bool = _interact(self, other_interactor)
            if interaction_bool:
                return_interactions.append(
                    _reportInteract(interaction_bool, self, other_interactor)
                )

            # Test for an inverse interaction
            inverse_interaction_bool = _interact(other_interactor, self)
            if inverse_interaction_bool:
                return_interactions.append(
                    _reportInteract(inverse_interaction_bool, other_interactor, self)
                )

            return return_interactions

        # Create a list to store the results
        interaction_list = []

        # Iterate the interaction
        for interaction in self._interactions:
            interaction_list.extend(_processInteraction(**interaction))

        return interaction_list

    @classmethod
    def withModelDict(cls, name, frame, nodes, coords_array, model_dict, **kwargs):
        return cls(
            name=name,
            frame=frame,
            nodes=nodes,
            coords_array=coords_array,
            skeleton=model_dict["Skeleton"],
            model=model_dict["Models"],
            interactions=model_dict["Interactions"],
            **kwargs,
        )
