# LSST-satellite-tracing

Low-earth satellites, such as Starlink constellation launched by SpaceX, have the potential to
significantly affect Rubin Observatory’s Legacy Survey of Space and Time (LSST). When a satellite crosses
the Rubin field of view during an exposure, it leaves a bright trail that can even saturate pixels. The
presence of trails affects ability to detect objects and measure their properties and can introduce
systematic effects that can limit scientific analysis. In order to quantitatively assess the magnitude
of this problem, high-fidelity simulations of satellite constellations are required.

This code can generate a user-specified satellite constellation, including the motion of all satellites,
and find all CCDs and pixels in the LSST field-of view that a trail intersects, provided the location
and duration of an LSST exposure.
