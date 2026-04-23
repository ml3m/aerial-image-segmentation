(function () {
  var form = document.getElementById("infer-form");
  if (!form) return;

  var dropzone = document.getElementById("dropzone");
  var input = document.getElementById("file-input");
  var maxMb = parseInt(form.getAttribute("data-max-mb") || "50", 10);
  var maxBytes = maxMb * 1024 * 1024;

  function setError(msg) {
    var old = form.querySelector(".js-client-error");
    if (old) old.remove();
    if (!msg) return;
    var div = document.createElement("div");
    div.className = "banner banner-error js-client-error";
    div.setAttribute("role", "alert");
    div.textContent = msg;
    form.insertBefore(div, form.firstChild);
  }

  function validateFile(file) {
    if (!file) {
      setError("Please choose an image file.");
      return false;
    }
    if (file.size > maxBytes) {
      setError("File exceeds maximum size of " + maxMb + " MB.");
      return false;
    }
    var ok = /\.(png|jpe?g|tiff?)$/i.test(file.name);
    if (!ok) {
      setError("Allowed types: PNG, JPEG, TIFF.");
      return false;
    }
    setError(null);
    return true;
  }

  dropzone.addEventListener("click", function () {
    input.click();
  });

  dropzone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });

  input.addEventListener("change", function () {
    var file = input.files && input.files[0];
    if (file) {
      var text = dropzone.querySelector(".dropzone-text");
      if (text) text.textContent = "Selected: " + file.name;
    }
  });

  ["dragenter", "dragover"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", function (e) {
    var dt = e.dataTransfer;
    if (!dt || !dt.files || !dt.files.length) return;
    var file = dt.files[0];
    try {
      input.files = dt.files;
    } catch (err) {
      setError("Could not attach dropped file. Use the file picker instead.");
      return;
    }
    var text = dropzone.querySelector(".dropzone-text");
    if (text) text.textContent = "Selected: " + file.name;
    setError(null);
  });

  form.addEventListener("submit", function (e) {
    var file = input.files && input.files[0];
    if (!validateFile(file)) {
      e.preventDefault();
    }
  });
})();
