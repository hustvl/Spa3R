from .base import BaseGeometryEncoder
from .vggt_encoder import VGGTEncoder
from .spa3r_encoder import Spa3REncoder


def create_geometry_encoder(config) -> BaseGeometryEncoder:
    """
    Factory function to create geometry encoders.
    
    Args:
        config: GeometryEncoderConfig instance with encoder configuration.
    Returns:
        Geometry encoder instance
    """

    encoder_type = config.encoder_type.lower()
    if encoder_type == "vggt":
        return VGGTEncoder(config)
    elif encoder_type == "spa3r":
        return Spa3REncoder(config)
    else:
        raise ValueError(f"Unknown geometry encoder type: {encoder_type}")


def get_available_encoders():
    """Get list of available encoder types."""
    return ["vggt", "spa3r"]
