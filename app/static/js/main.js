document.addEventListener('DOMContentLoaded', function() {
    // Inisialisasi SocketIO
    // Ganti dengan namespace yang benar jika Anda menggunakannya di server
    const socket = io(); // Jika server SocketIO di root yang sama
    // const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);


    socket.on('connect', () => {
        console.log('Terhubung ke server SocketIO');
        // Jika kita berada di halaman pemrosesan, kirim event untuk join room
        const analysisIdElement = document.getElementById('analysisId');
        if (analysisIdElement) {
            const analysisId = analysisIdElement.value;
            if (analysisId) {
                socket.emit('join_room', { analysis_id: analysisId });
                console.log('Mengirim permintaan join_room untuk analysis_id:', analysisId);
            }
        }
    });

    socket.on('disconnect', () => {
        console.log('Terputus dari server SocketIO');
    });

    socket.on('initial_status', (data) => {
        console.log('Menerima status awal:', data);
        // Update UI dengan status awal jika perlu
        const statusMessageDiv = document.getElementById('statusMessage');
        if (statusMessageDiv) {
            statusMessageDiv.textContent = `Status Awal: ${data.message}`;
        }
    });

    socket.on('progress_update', (data) => {
        console.log('Menerima progress_update:', data);
        
        const progressBar = document.getElementById('progressBar');
        const statusMessage = document.getElementById('statusMessage');
        const currentStepDisplay = document.getElementById('currentStepDisplay'); // Untuk nama langkah
        const nextStepButton = document.getElementById('nextStepButton');
        const viewResultsButton = document.getElementById('viewResultsButton');

        if (progressBar) {
            progressBar.style.width = data.progress + '%';
            progressBar.textContent = data.progress + '%';
        }
        if (statusMessage) {
            let message = `Langkah: ${data.step}, Status: ${data.status}`;
            if (data.error) {
                message += `, Error: ${data.error}`;
                if(progressBar) progressBar.style.backgroundColor = 'red';
            } else if (data.status === 'selesai' && progressBar) {
                 progressBar.style.backgroundColor = '#28a745'; // Hijau untuk sukses
            }
            statusMessage.textContent = message;
        }
        if (currentStepDisplay && data.step) {
            // Anda mungkin perlu memetakan nama internal langkah ke nama tampilan
            currentStepDisplay.textContent = `Memproses: ${data.step}`;
        }

        // Update status langkah di sidebar jika ada
        const stepLiElement = document.getElementById(`step-li-${data.step_id |
| data.step}`); // step_id atau nama internal
        if (stepLiElement) {
            if (data.status === 'selesai') {
                stepLiElement.classList.remove('active-step');
                stepLiElement.classList.add('completed-step');
                // Aktifkan langkah berikutnya di sidebar
                const nextStepId = parseInt(stepLiElement.dataset.stepId) + 1;
                const nextStepLi = document.getElementById(`step-li-${nextStepId}`);
                if(nextStepLi) nextStepLi.classList.add('active-step');

            } else if (data.status === 'gagal') {
                stepLiElement.classList.add('failed-step'); // Tambahkan class CSS untuk gagal
                stepLiElement.style.color = 'red';
            }
        }


        // Jika langkah saat ini selesai dan ada langkah berikutnya
        if (data.status === 'selesai') {
            const analysisId = document.getElementById('analysisId').value;
            const currentStepId = parseInt(document.getElementById('currentStepId').value);
            const totalSteps = parseInt(document.getElementById('totalSteps').value);

            if (currentStepId < totalSteps) {
                if (nextStepButton) {
                    nextStepButton.disabled = false;
                    nextStepButton.textContent = `Lanjut ke Langkah ${currentStepId + 1}`;
                    // Update action URL form jika perlu, atau biarkan server yang mengarahkan
                    const form = document.getElementById('processingForm');
                    if(form) {
                        form.action = `/analysis/process/${analysisId}/step/${currentStepId + 1}`;
                    }
                }
            } else { // Langkah terakhir selesai
                if (nextStepButton) nextStepButton.style.display = 'none'; // Sembunyikan tombol lanjut
                if (viewResultsButton) {
                    viewResultsButton.style.display = 'inline-block'; // Tampilkan tombol lihat hasil
                    viewResultsButton.href = `/analysis/results/${analysisId}`;
                }
                if (statusMessage) statusMessage.textContent = "Semua langkah analisis telah selesai!";
            }
        } else if (data.status === 'gagal') {
            if (nextStepButton) {
                nextStepButton.disabled = true;
                nextStepButton.textContent = "Analisis Gagal";
            }
        }


    });

    // Validasi form unggah sisi klien (opsional, server tetap harus validasi)
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            const fileInput = document.getElementById('file');
            const submitButton = uploadForm.querySelector('button[type="submit"]');
            if (fileInput.files.length === 0) {
                alert('Silakan pilih file untuk diunggah.');
                event.preventDefault();
                return;
            }
            // Tambahkan validasi ekstensi atau ukuran jika perlu
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.textContent = 'Mengunggah...';
            }
        });
    }

    // Logika untuk tombol "Lanjut ke Langkah Berikutnya" di halaman processing_step.html
    // Ini akan submit form yang kemudian memicu task Celery baru
    const processingForm = document.getElementById('processingForm');
    if (processingForm) {
        const nextButton = processingForm.querySelector('#nextStepButton');
        if (nextButton && nextButton.tagName === 'BUTTON' && nextButton.type === 'submit') {
             // Jika tombol "Lanjut" adalah tombol submit, tidak perlu event listener khusus
             // kecuali untuk menonaktifkannya saat submit
            processingForm.addEventListener('submit', function() {
                if (nextButton) {
                    nextButton.disabled = true;
                    nextButton.textContent = 'Memproses...';
                }
                // Kosongkan progress bar dan status message untuk langkah baru
                const progressBar = document.getElementById('progressBar');
                const statusMessage = document.getElementById('statusMessage');
                if(progressBar) {
                    progressBar.style.width = '0%';
                    progressBar.textContent = '0%';
                    progressBar.style.backgroundColor = '#4A5DB5'; // Reset warna
                }
                if(statusMessage) {
                    statusMessage.textContent = 'Memulai langkah baru...';
                }
            });
        }
    }

    // Menangani klik pada item sidebar untuk navigasi (jika diizinkan)
    const stepLinks = document.querySelectorAll('.step-sidebar-item a');
    stepLinks.forEach(link => {
        link.addEventListener('click', function(event) {
            // Anda mungkin ingin menambahkan logika di sini untuk memeriksa apakah
            // pengguna diizinkan untuk melompat ke langkah tersebut (misalnya, jika sudah selesai)
            // Untuk saat ini, biarkan navigasi standar terjadi.
            // Jika langkah belum selesai, server akan mengarahkan kembali atau menampilkan pesan.
            console.log(`Navigasi ke: ${this.href}`);
        });
    });


});