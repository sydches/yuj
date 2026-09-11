"""Give each pytest invocation an owned report and observed collection root."""
import os
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    expected = os.environ.get('YUJ_NATIVE_REPORT')
    if not expected:
        return
    path = Path(expected + '.' + uuid.uuid4().hex + '.xml')
    path.touch(exist_ok=False)  # An unfinished invocation leaves an empty report.
    if str(getattr(config.option, 'xmlpath', None)) != expected:
        return  # Preserve redirection, but record incomplete owned coverage.
    config.option.xmlpath = str(path)
    config._yuj_native_report = path


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    yield  # Let the native JUnit writer finish first.
    path = getattr(session.config, '_yuj_native_report', None)
    if path is None:
        return
    try:
        report = ET.parse(path)
        report.getroot().set('yuj_collection_root', str(session.config.rootpath.resolve()))
        report.write(path, encoding='utf-8', xml_declaration=True)
    except (OSError, ET.ParseError):
        pass  # The collector keeps incomplete evidence unavailable.
