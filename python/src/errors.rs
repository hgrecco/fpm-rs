use fpm_rs::Error;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

create_exception!(fpm_rs, FpmError, PyException);
create_exception!(fpm_rs, InvalidShapeError, FpmError);
create_exception!(fpm_rs, InvalidParameterError, FpmError);
create_exception!(fpm_rs, InvalidModelError, FpmError);
create_exception!(fpm_rs, InvalidMeasurementsError, FpmError);
create_exception!(fpm_rs, LengthMismatchError, FpmError);
create_exception!(fpm_rs, FrameOutOfRangeError, FpmError);
create_exception!(fpm_rs, NumericalError, FpmError);
create_exception!(fpm_rs, UnsupportedError, FpmError);
create_exception!(fpm_rs, DatasetError, FpmError);
create_exception!(fpm_rs, FpmIoError, FpmError);
create_exception!(fpm_rs, SerializationError, FpmError);

pub(crate) fn to_py_err(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::InvalidShape(_) | Error::ShapeOverflow { .. } | Error::NdarrayShape(_) => {
            InvalidShapeError::new_err(message)
        }
        Error::InvalidParameter { .. } => InvalidParameterError::new_err(message),
        Error::InvalidModel(_) => InvalidModelError::new_err(message),
        Error::InvalidMeasurements(_) => InvalidMeasurementsError::new_err(message),
        Error::LengthMismatch { .. } | Error::ArrayLengthMismatch { .. } => {
            LengthMismatchError::new_err(message)
        }
        Error::NonStandardLayout { .. } => InvalidShapeError::new_err(message),
        Error::FrameOutOfRange { .. } => FrameOutOfRangeError::new_err(message),
        Error::Numerical(_) => NumericalError::new_err(message),
        Error::Unsupported(_) => UnsupportedError::new_err(message),
        Error::Dataset(_) => DatasetError::new_err(message),
        Error::Io(_) | Error::Csv(_) | Error::Image(_) | Error::Tiff(_) => {
            FpmIoError::new_err(message)
        }
        Error::Serialization(_) => SerializationError::new_err(message),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("FpmError", module.py().get_type::<FpmError>())?;
    module.add(
        "InvalidShapeError",
        module.py().get_type::<InvalidShapeError>(),
    )?;
    module.add(
        "InvalidParameterError",
        module.py().get_type::<InvalidParameterError>(),
    )?;
    module.add(
        "InvalidModelError",
        module.py().get_type::<InvalidModelError>(),
    )?;
    module.add(
        "InvalidMeasurementsError",
        module.py().get_type::<InvalidMeasurementsError>(),
    )?;
    module.add(
        "LengthMismatchError",
        module.py().get_type::<LengthMismatchError>(),
    )?;
    module.add(
        "FrameOutOfRangeError",
        module.py().get_type::<FrameOutOfRangeError>(),
    )?;
    module.add("NumericalError", module.py().get_type::<NumericalError>())?;
    module.add(
        "UnsupportedError",
        module.py().get_type::<UnsupportedError>(),
    )?;
    module.add("DatasetError", module.py().get_type::<DatasetError>())?;
    module.add("FpmIoError", module.py().get_type::<FpmIoError>())?;
    module.add(
        "SerializationError",
        module.py().get_type::<SerializationError>(),
    )?;
    Ok(())
}
