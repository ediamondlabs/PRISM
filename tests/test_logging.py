import logging


from src.utils.logging_utils import log_config, setup_logger


def test_setup_logger_default():
    """Test logger setup with default parameters."""
    logger = setup_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1


def test_setup_logger_with_file(tmp_path):
    """Test logger setup with file handler."""
    log_file = tmp_path / "test.log"
    logger = setup_logger("test_logger_file", log_file=log_file)

    assert len(logger.handlers) == 2
    assert log_file.exists()

    logger.info("Test message")
    content = log_file.read_text()
    assert "Test message" in content


def test_setup_logger_custom_level():
    """Test logger setup with custom logging level."""
    logger = setup_logger("test_logger_debug", level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_setup_logger_custom_format():
    """Test logger setup with custom format string."""
    custom_format = "%(levelname)s - %(message)s"
    logger = setup_logger("test_logger_format", format_string=custom_format)
    assert len(logger.handlers) >= 1


def test_log_config(caplog):
    """Test configuration logging."""
    logger = setup_logger("test_logger_config")

    test_config = {
        "model": {"hidden_size": 128, "num_layers": 2},
        "training": {"learning_rate": 0.001, "batch_size": 32},
        "seed": 42,
    }

    with caplog.at_level(logging.INFO):
        log_config(logger, test_config)

    assert "Configuration:" in caplog.text
    assert "hidden_size: 128" in caplog.text
    assert "learning_rate: 0.001" in caplog.text
    assert "seed: 42" in caplog.text


def test_log_config_nested_dict(caplog):
    """Test logging of nested configuration dictionaries."""
    logger = setup_logger("test_logger_nested")

    nested_config = {
        "level1": {
            "level2": {"level3": {"value": 123}},
            "other_value": "test",
        }
    }

    with caplog.at_level(logging.INFO):
        log_config(logger, nested_config)

    assert "level1:" in caplog.text
    assert "level2:" in caplog.text
    assert "level3:" in caplog.text
    assert "value: 123" in caplog.text


def test_logger_file_creation_in_nested_dirs(tmp_path):
    """Test that logger creates nested directories for log file."""
    log_file = tmp_path / "nested" / "dirs" / "test.log"
    logger = setup_logger("test_logger_nested_dirs", log_file=log_file)

    logger.info("Test message")

    assert log_file.exists()
    assert log_file.parent.exists()


def test_logger_clears_previous_handlers():
    """Test that logger clears previous handlers on re-initialization."""
    logger = setup_logger("test_logger_clear")
    initial_handlers = len(logger.handlers)

    logger = setup_logger("test_logger_clear")
    assert len(logger.handlers) == initial_handlers
