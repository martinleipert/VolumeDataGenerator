import random
import numpy
from skimage.morphology import ball, cube
import re
from skimage.util import random_noise
from skimage.filters import gaussian


class RandomMorphologyGenerator:

    def __init__(self, volume_size, num_objects):
        self.__volume_size = volume_size
        self.__num_objects = num_objects

        self.__width_range = (24, 48)
        self.primitives = [ball, cube]
        self.num_primitives = len(self.primitives)

    def __iter__(self):
        pass

    def generate_samples(self, num_samples=1):

        volumes = numpy.zeros((num_samples, self.__volume_size[0], self.__volume_size[1], self.__volume_size[2]))
        annotations = []
        targets = []

        for sample_index in range(num_samples):

            volume = numpy.zeros(self.__volume_size).astype(numpy.float32)
            labels = numpy.zeros_like(volume).astype(numpy.int32)

            annotation = {
                "boxes": [],
                "centers": [],
                "masks": [],
                "classes": []
            }

            for object_index in range(self.__num_objects):

                primitive_idx = random.randint(0, self.num_primitives - 1)
                primitive = self.primitives[primitive_idx]
                class_label = primitive_idx + 1

                width = random.randint(self.__width_range[0], self.__width_range[1])

                if primitive is ball:
                    radius = (width / 2)
                else:
                    radius = width

                min_pose = numpy.array([width, width, width]).astype(numpy.int32) + 1
                max_pose = numpy.subtract(self.__volume_size, width) - 1

                center = random.randint(min_pose[0], max_pose[0]), \
                    random.randint(min_pose[1], max_pose[1]), \
                    random.randint(min_pose[2], max_pose[2])

                mask = primitive(radius)

                min_loc = numpy.subtract(center, width).astype(numpy.int32)
                max_loc = numpy.add(min_loc, mask.shape).astype(numpy.int32)

                volume[min_loc[0]:max_loc[0], min_loc[1]:max_loc[1], min_loc[2]:max_loc[2]] = \
                    numpy.where(mask, 1, volume[min_loc[0]:max_loc[0], min_loc[1]:max_loc[1], min_loc[2]:max_loc[2]])
                labels[min_loc[0]:max_loc[0], min_loc[1]:max_loc[1], min_loc[2]:max_loc[2]] = \
                    numpy.where(mask, class_label,
                                volume[min_loc[0]:max_loc[0], min_loc[1]:max_loc[1], min_loc[2]:max_loc[2]])

                shape = mask.shape

                annotation["boxes"].append((min_loc[0], min_loc[1], min_loc[2], min_loc[0] + shape[0],
                                            min_loc[1] + shape[1], min_loc[2] + shape[2]))
                annotation["classes"].append(class_label)
                annotation["centers"].append(center)
                annotation["masks"].append(mask)

            volume = gaussian(volume)
            volume = random_noise(volume, var=1e-3)

            volumes[sample_index] = volume
            annotations.append(annotation)

            # target = BoxList(annotation["boxes"], volume.shape)
            # target.add_field("labels", annotation["classes"])
            # target.add_field("masks", annotation["masks"])
            # target.add_field("centers", annotation["centers"])
            # target.add_field("classes", annotation["classes"])
            # targets.append(target)

        return volumes, annotations, targets
