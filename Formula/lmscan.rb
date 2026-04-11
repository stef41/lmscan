class Lmscan < Formula
  include Language::Python::Virtualenv

  desc "AI text forensics — detect AI-generated text and fingerprint which LLM wrote it"
  homepage "https://github.com/stef41/lmscan"
  url "https://files.pythonhosted.org/packages/source/l/lmscan/lmscan-0.3.0.tar.gz"
  sha256 "a89443ba03a3a581e3b1608ce5a16f245c5b4e8ea642fc361c9827bd159e5a59"
  license "Apache-2.0"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    output = shell_output("#{bin}/lmscan 'Hello world'")
    assert_match "probability", output
  end
end
