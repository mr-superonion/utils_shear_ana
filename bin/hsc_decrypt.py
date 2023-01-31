#!/usr/bin/env python
import os
import sys
import gnupg
import getpass

import astropy.io.fits as pyfits

# NOTE: Make sure it is the gpg directory
# And the gpg contains the private key
gpgfile = os.path.join(os.environ["HOME"], ".gnupg")
gpg = gnupg.GPG(gnupghome=gpgfile)


def PGP_to_FITS(message):
    """
    Turns a PGP message into a string that can be stored in a FITS
    header, by removing the header and footer of the message as well
    as any \n characters
    """
    s = ""
    for st in message.split("\n"):
        if len(st) > 0 and "PGP MESSAGE" not in st and "GnuPG" not in st:
            s += st
    return s


def FITS_to_PGP(message):
    """
    Turns a string stored into an FITS comment back into a proper
    PGP message
    """
    s = "-----BEGIN PGP MESSAGE-----\n\n"
    s += message
    s += "\n-----END PGP MESSAGE-----\n"
    return s


def decrypt(header):
    n_blind_cat = header["ncat"]
    keys = gpg.list_keys(secret=True)
    private_fingerprints = [k["fingerprint"] for k in keys]
    keyid1 = header["uid"]
    if keyid1 in private_fingerprints:
        print("Decrypt User Blinding:")
        pwd = getpass.getpass(prompt="Password for dm1: ")
        for ib in range(n_blind_cat):
            dm1_cry = FITS_to_PGP(header["dm1c%s" % ib])
            dec_s1 = gpg.decrypt(dm1_cry, passphrase=pwd)
            assert dec_s1.ok, "Problem during decryption: %s" % dec_s1.stderr
            dm = eval(dec_s1.data)
            print("dm1-cat%d: %.5f" % (ib, dm))
    else:
        print("Cannot find user's fingerprint to decrypt dm1!!")

    if n_blind_cat != 3:
        print("Do not have HSC level blinding")
        return
    keyid2 = header["bid"]
    if keyid2 in private_fingerprints:
        print("\nDecrypt HSC Blinding:")
        pwd = getpass.getpass(prompt="Password for dm2: ")

        seed_cry = FITS_to_PGP(header["seed"])
        dec_s2 = gpg.decrypt(seed_cry, passphrase=pwd)
        assert dec_s2.ok, "Problem during decryption: %s" % dec_s2.stderr
        seed = eval(dec_s2.data)
        print("Random seed is: %d" % seed)
        for ib in range(n_blind_cat):
            dm2_cry = FITS_to_PGP(header["dm2c%s" % ib])
            dec_s2 = gpg.decrypt(dm2_cry, passphrase=pwd)
            assert dec_s2.ok, "Problem during decryption: %s" % dec_s2.stderr
            dm = eval(dec_s2.data)
            print("dm2-cat%d: %.5f" % (ib, dm))
    else:
        print("Cannot find hsc's  fingerprint for dm2!!")
    return


def main(argv):
    if len(argv) == 2:
        if os.path.isfile(argv[1]):
            decrypt(pyfits.getheader(argv[1], 1))
        else:
            print("Cannot find the fits file %s" % argv[2])
    elif len(argv) < 2:
        print("Please input the relative directory of a fits file !!")
    elif len(argv) > 2:
        print("Wrong number of inputs !!")


if __name__ == "__main__":
    main(sys.argv)
