#pragma once

#define LINTDB_VERSION_STRING "0.3.0"


namespace lintdb {
    struct Version
    {
        Version() = default;
        Version(string versionStr)
        {
            sscanf(versionStr.c_str(), "%d.%d.%d", &major, &minor, &revision);
            metadata_enabled = major >= 0 && minor >= 3 && revision >= 0;
        }

        bool operator<(const Version &otherVersion)
        {
            if(major < otherVersion.major)
                return true;
            if(minor < otherVersion.minor)
                return true;
            if(revision < otherVersion.revision)
                return true;
            return false;
        }

        bool metadata_enabled;

        int major, minor, revision, build;
    };

    static const Version LINTDB_VERSION(LINTDB_VERSION_STRING);
}

