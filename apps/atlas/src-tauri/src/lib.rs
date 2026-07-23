use std::{
    env, fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};

use rand::RngCore;
use serde::Serialize;
use tauri::{webview::PageLoadEvent, AppHandle, Manager, RunEvent, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const BACKEND_HOST: &str = "127.0.0.1";
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const BACKEND_START_ATTEMPTS: usize = 5;
const BACKEND_STARTUP_ATTEMPTS: usize = 80;
const BACKEND_STARTUP_RETRY_MS: u64 = 250;
const BACKEND_GRACEFUL_SHUTDOWN_TIMEOUT_MS: u64 = 12_000;
const BACKEND_SHUTDOWN_POLL_MS: u64 = 100;
const WINDOW_REVEAL_FALLBACK_MS: u64 = 500;

#[derive(Clone, Serialize)]
struct BackendRuntime {
    host: String,
    port: u16,
    token: String,
}

#[derive(Clone, Serialize)]
struct AppDiagnostics {
    platform: String,
    data_dir: String,
    log_dir: String,
    backend_log_path: String,
    packaged_logs_enabled: bool,
}

#[derive(Default)]
struct BackendState {
    child: Mutex<Option<Child>>,
    runtime: Mutex<Option<BackendRuntime>>,
}

#[tauri::command]
fn restart_backend(app: AppHandle, state: State<'_, BackendState>) -> Result<(), String> {
    stop_backend(&state)?;
    start_backend(app, &state)
}

#[tauri::command]
fn backend_runtime(state: State<'_, BackendState>) -> Result<BackendRuntime, String> {
    let guard = state
        .runtime
        .lock()
        .map_err(|_| "Backend runtime lock is poisoned.".to_string())?;
    guard
        .clone()
        .ok_or_else(|| "Atlas backend runtime is not available.".to_string())
}

#[tauri::command]
fn open_external_url(url: String) -> Result<(), String> {
    if is_allowed_external_url(&url) {
        open_allowed_external_url(&url)
    } else {
        Err("External URL is not allowed.".to_string())
    }
}

#[tauri::command]
fn app_diagnostics(app: AppHandle) -> Result<AppDiagnostics, String> {
    let data_dir = atlas_data_dir(&app)?;
    let log_dir = data_dir.join("logs");
    let backend_log_path = log_dir.join("backend.log");
    Ok(AppDiagnostics {
        platform: env::consts::OS.to_string(),
        data_dir: data_dir.to_string_lossy().to_string(),
        log_dir: log_dir.to_string_lossy().to_string(),
        backend_log_path: backend_log_path.to_string_lossy().to_string(),
        packaged_logs_enabled: cfg!(debug_assertions) || packaged_backend_logs_enabled(),
    })
}

#[tauri::command]
fn open_app_location(app: AppHandle, location: String) -> Result<(), String> {
    let path = match location.as_str() {
        "data" => atlas_data_dir(&app)?,
        "logs" => atlas_data_dir(&app)?.join("logs"),
        _ => return Err("Atlas location is not allowed.".to_string()),
    };
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    open_local_path(&path)
}

fn is_allowed_external_url(url: &str) -> bool {
    matches!(
        url,
        "https://ollama.com/download" | "https://github.com/olliedoganay/AtlasChat"
    )
}

pub fn run() {
    tauri::Builder::default()
        .manage(BackendState::default())
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .on_page_load(|webview, payload| {
            if webview.label() == "main" && payload.event() == PageLoadEvent::Finished {
                if let Some(window) = webview.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.unminimize();
                    let _ = window.set_focus();
                }
            }
        })
        .setup(|app| {
            start_backend(app.handle().clone(), &app.state::<BackendState>())?;
            reveal_main_window_after_delay(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            restart_backend,
            backend_runtime,
            open_external_url,
            app_diagnostics,
            open_app_location
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| match event {
            RunEvent::ExitRequested { .. } | RunEvent::Exit => {
                let state = app.state::<BackendState>();
                let _ = stop_backend(&state);
            }
            _ => {}
        });
}

fn open_allowed_external_url(url: &str) -> Result<(), String> {
    #[cfg(windows)]
    {
        Command::new("rundll32.exe")
            .args(["url.dll,FileProtocolHandler", url])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|error| format!("Could not open external URL: {error}"))?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(url)
            .spawn()
            .map_err(|error| format!("Could not open external URL: {error}"))?;
        return Ok(());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(url)
            .spawn()
            .map_err(|error| format!("Could not open external URL: {error}"))?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err("Opening external URLs is not supported on this platform.".to_string())
}

fn open_local_path(path: &Path) -> Result<(), String> {
    #[cfg(windows)]
    {
        Command::new("explorer.exe")
            .arg(path)
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|error| format!("Could not open Atlas location: {error}"))?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|error| format!("Could not open Atlas location: {error}"))?;
        return Ok(());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|error| format!("Could not open Atlas location: {error}"))?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err("Opening local Atlas locations is not supported on this platform.".to_string())
}

fn reveal_main_window_after_delay(app: AppHandle) {
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(WINDOW_REVEAL_FALLBACK_MS));
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.show();
            let _ = window.unminimize();
            let _ = window.set_focus();
        }
    });
}

fn start_backend(app: AppHandle, state: &State<'_, BackendState>) -> Result<(), String> {
    let mut guard = state
        .child
        .lock()
        .map_err(|_| "Backend lock is poisoned.".to_string())?;

    if let Some(child) = guard.as_mut() {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            return Ok(());
        }
        *guard = None;
    }

    let repo_root = repo_root()?;
    let mut last_error = "Atlas backend exited during startup.".to_string();
    for attempt in 0..BACKEND_START_ATTEMPTS {
        let runtime = BackendRuntime {
            host: BACKEND_HOST.to_string(),
            port: reserve_port()?,
            token: generate_instance_token(),
        };
        let (program, args, launch_mode) = backend_command(&app, &repo_root, &runtime)?;
        let mut command = Command::new(program);
        command.args(args).stdin(Stdio::null());

        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);

        match launch_mode {
            LaunchMode::Development => {
                let log_dir = repo_root.join(".data").join("logs");
                fs::create_dir_all(&log_dir).map_err(|error| error.to_string())?;
                let (stdout, stderr) = backend_log_streams(&log_dir.join("backend.log"))?;
                command.stdout(stdout).stderr(stderr);
                command
                    .current_dir(&repo_root)
                    .env("ATLAS_API_HOST", BACKEND_HOST)
                    .env("ATLAS_API_PORT", runtime.port.to_string())
                    .env("ATLAS_INSTANCE_TOKEN", &runtime.token)
                    .env("MEM0_DIR", repo_root.join(".data").join("mem0"));

                if let Some(path) = playwright_browsers_path() {
                    command.env("PLAYWRIGHT_BROWSERS_PATH", path);
                }

                let python_path = repo_root.join("src");
                let mut python_paths = vec![python_path];
                if let Some(existing_python_path) = env::var_os("PYTHONPATH") {
                    python_paths.extend(env::split_paths(&existing_python_path));
                }
                let merged_python_path =
                    env::join_paths(python_paths).map_err(|error| error.to_string())?;
                command.env("PYTHONPATH", merged_python_path);
            }
            LaunchMode::Packaged {
                resource_dir,
                data_dir,
            } => {
                fs::create_dir_all(&data_dir).map_err(|error| error.to_string())?;
                fs::create_dir_all(data_dir.join("langgraph"))
                    .map_err(|error| error.to_string())?;
                let (stdout, stderr) = if packaged_backend_logs_enabled() {
                    let log_dir = data_dir.join("logs");
                    fs::create_dir_all(&log_dir).map_err(|error| error.to_string())?;
                    backend_log_streams(&log_dir.join("backend.log"))?
                } else {
                    (Stdio::null(), Stdio::null())
                };
                command.stdout(stdout).stderr(stderr);
                command
                    .current_dir(&data_dir)
                    .env("ATLAS_API_HOST", BACKEND_HOST)
                    .env("ATLAS_API_PORT", runtime.port.to_string())
                    .env("ATLAS_INSTANCE_TOKEN", &runtime.token)
                    .env("ATLAS_PROJECT_ROOT", &resource_dir)
                    .env("ATLAS_PROMPT_DIR", resource_dir.join("prompts"))
                    .env("ATLAS_DATA_DIR", &data_dir)
                    .env("MEM0_DIR", data_dir.join("mem0"))
                    .env("QDRANT_PATH", data_dir.join("qdrant"))
                    .env(
                        "LANGGRAPH_CHECKPOINT_DB",
                        data_dir.join("langgraph").join("checkpoints.sqlite"),
                    )
                    .env("WORLD_DB_PATH", data_dir.join("world.sqlite"))
                    .env("BROWSER_STORAGE_DIR", data_dir.join("browser_runs"))
                    .env("BENCHMARKS_DIR", data_dir.join("benchmarks"))
                    .env("EVALS_DIR", data_dir.join("evals"))
                    .env("PROPOSALS_DIR", data_dir.join("profiles"))
                    .env("MEM0_HISTORY_DB", data_dir.join("mem0_history.sqlite"));

                if let Some(path) = playwright_browsers_path() {
                    command.env("PLAYWRIGHT_BROWSERS_PATH", path);
                }
            }
        }

        let mut child = command.spawn().map_err(|error| error.to_string())?;
        if !wait_for_backend_ready(&runtime, &mut child)? {
            last_error = format!(
                "Atlas backend did not become ready on port {}.",
                runtime.port
            );
            if child
                .try_wait()
                .map_err(|error| error.to_string())?
                .is_none()
            {
                let _ = child.kill();
            }
            let _ = child.wait();
            if attempt + 1 < BACKEND_START_ATTEMPTS {
                continue;
            }
            return Err(last_error);
        }
        *guard = Some(child);
        let mut runtime_guard = state
            .runtime
            .lock()
            .map_err(|_| "Backend runtime lock is poisoned.".to_string())?;
        *runtime_guard = Some(runtime);
        return Ok(());
    }
    Err(last_error)
}

fn stop_backend(state: &State<'_, BackendState>) -> Result<(), String> {
    let runtime = state
        .runtime
        .lock()
        .map_err(|_| "Backend runtime lock is poisoned.".to_string())?
        .clone();
    if let Some(runtime) = runtime.as_ref() {
        let _ = request_backend_shutdown(runtime);
    }

    let mut guard = state
        .child
        .lock()
        .map_err(|_| "Backend lock is poisoned.".to_string())?;

    if let Some(mut child) = guard.take() {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            let exited_gracefully = if runtime.is_some() {
                wait_for_child_exit(
                    &mut child,
                    Duration::from_millis(BACKEND_GRACEFUL_SHUTDOWN_TIMEOUT_MS),
                )?
            } else {
                false
            };
            if !exited_gracefully {
                child.kill().map_err(|error| error.to_string())?;
                let _ = child.wait();
            }
        }
    }
    let mut runtime_guard = state
        .runtime
        .lock()
        .map_err(|_| "Backend runtime lock is poisoned.".to_string())?;
    *runtime_guard = None;
    Ok(())
}

fn wait_for_backend_ready(runtime: &BackendRuntime, child: &mut Child) -> Result<bool, String> {
    for _ in 0..BACKEND_STARTUP_ATTEMPTS {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some()
        {
            return Ok(false);
        }
        if authenticated_health_check(runtime) {
            return Ok(true);
        }
        thread::sleep(Duration::from_millis(BACKEND_STARTUP_RETRY_MS));
    }
    Ok(false)
}

fn authenticated_health_check(runtime: &BackendRuntime) -> bool {
    let address = format!("{}:{}", runtime.host, runtime.port);
    let Ok(socket_address) = address.parse() else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&socket_address, Duration::from_millis(200))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
    let request = format!(
        "GET /health HTTP/1.1\r\nHost: {}\r\nX-Atlas-Instance-Token: {}\r\nConnection: close\r\n\r\n",
        address, runtime.token
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = [0_u8; 96];
    match stream.read(&mut response) {
        Ok(read) if read > 0 => String::from_utf8_lossy(&response[..read]).contains(" 200 "),
        _ => false,
    }
}

fn wait_for_child_exit(child: &mut Child, timeout: Duration) -> Result<bool, String> {
    let deadline = Instant::now() + timeout;
    loop {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some()
        {
            return Ok(true);
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(false);
        }
        thread::sleep(
            Duration::from_millis(BACKEND_SHUTDOWN_POLL_MS)
                .min(deadline.saturating_duration_since(now)),
        );
    }
}

fn request_backend_shutdown(runtime: &BackendRuntime) -> bool {
    let address = format!("{}:{}", runtime.host, runtime.port);
    let Ok(socket_address) = address.parse() else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&socket_address, Duration::from_millis(350))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(1500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
    let request = prepare_shutdown_request(&address, &runtime.token);
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = [0_u8; 96];
    match stream.read(&mut response) {
        Ok(read) if read > 0 => String::from_utf8_lossy(&response[..read]).contains(" 200 "),
        _ => false,
    }
}

fn prepare_shutdown_request(address: &str, token: &str) -> String {
    format!(
        "POST /admin/prepare-shutdown HTTP/1.1\r\nHost: {address}\r\nX-Atlas-Instance-Token: {token}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
}

fn backend_command(
    app: &AppHandle,
    repo_root: &Path,
    runtime: &BackendRuntime,
) -> Result<(PathBuf, Vec<String>, LaunchMode), String> {
    if !cfg!(debug_assertions) {
        if insecure_localhost_override_enabled() {
            return Err(
                "Atlas cannot start because ATLAS_ALLOW_INSECURE_LOCALHOST is enabled in this environment. Remove that override and launch Atlas again.".to_string(),
            );
        }
        if let Some((exe, resource_dir)) = packaged_sidecar(app) {
            let data_dir = app
                .path()
                .app_data_dir()
                .map_err(|error| error.to_string())?
                .join("runtime");
            return Ok((
                exe,
                vec![
                    "--host".to_string(),
                    runtime.host.clone(),
                    "--port".to_string(),
                    runtime.port.to_string(),
                ],
                LaunchMode::Packaged {
                    resource_dir,
                    data_dir,
                },
            ));
        }
        return Err("Packaged Atlas backend sidecar was not found.".to_string());
    }

    let candidates = if cfg!(windows) {
        vec![
            repo_root.join(".venv").join("Scripts").join("python.exe"),
            repo_root.join(".venv").join("Scripts").join("python"),
        ]
    } else {
        vec![
            repo_root.join(".venv").join("bin").join("python"),
            repo_root.join(".venv").join("bin").join("python3"),
        ]
    };
    let program = if let Some(repo_python) = candidates.into_iter().find(|path| path.exists()) {
        repo_python
    } else if cfg!(windows) {
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    };
    Ok((
        program,
        vec![
            "-m".to_string(),
            "atlas_local.api".to_string(),
            "--host".to_string(),
            runtime.host.clone(),
            "--port".to_string(),
            runtime.port.to_string(),
        ],
        LaunchMode::Development,
    ))
}

fn atlas_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    if cfg!(debug_assertions) {
        return Ok(repo_root()?.join(".data"));
    }
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("runtime"))
}

fn packaged_sidecar(app: &AppHandle) -> Option<(PathBuf, PathBuf)> {
    let resource_dir = app.path().resource_dir().ok()?;
    for asset_root in [resource_dir.clone(), resource_dir.join("resources")] {
        for binary_name in packaged_backend_binary_names() {
            let sidecar = asset_root.join("backend").join(binary_name);
            if sidecar.exists() {
                return Some((sidecar, asset_root));
            }
        }
    }
    None
}

fn packaged_backend_binary_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["atlas-backend.exe", "atlas-backend"]
    } else {
        &["atlas-backend", "atlas-backend.exe"]
    }
}

fn repo_root() -> Result<PathBuf, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(PathBuf::from)
        .ok_or_else(|| "Failed to resolve repository root for Atlas desktop.".to_string())
}

enum LaunchMode {
    Development,
    Packaged {
        resource_dir: PathBuf,
        data_dir: PathBuf,
    },
}

fn reserve_port() -> Result<u16, String> {
    let listener = TcpListener::bind((BACKEND_HOST, 0)).map_err(|error| error.to_string())?;
    let port = listener
        .local_addr()
        .map_err(|error| error.to_string())?
        .port();
    drop(listener);
    Ok(port)
}

fn generate_instance_token() -> String {
    let mut bytes = [0_u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let encoded = bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("atlas-{encoded}")
}

fn backend_log_streams(path: &Path) -> Result<(Stdio, Stdio), String> {
    let stdout = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
    Ok((Stdio::from(stdout), Stdio::from(stderr)))
}

fn packaged_backend_logs_enabled() -> bool {
    matches!(
        env::var("ATLAS_ENABLE_PACKAGED_LOGS"),
        Ok(value) if value == "1" || value.eq_ignore_ascii_case("true")
    )
}

fn insecure_localhost_override_enabled() -> bool {
    matches!(
        env::var("ATLAS_ALLOW_INSECURE_LOCALHOST"),
        Ok(value) if value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes") || value.eq_ignore_ascii_case("on")
    )
}

fn playwright_browsers_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("PLAYWRIGHT_BROWSERS_PATH") {
        let resolved = PathBuf::from(path);
        if resolved.exists() {
            return Some(resolved);
        }
    }

    env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .map(|path| path.join("ms-playwright"))
        .filter(|path| path.exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graceful_shutdown_timeout_stays_within_desktop_exit_budget() {
        assert!((10_000..=15_000).contains(&BACKEND_GRACEFUL_SHUTDOWN_TIMEOUT_MS));
    }

    #[test]
    fn shutdown_request_targets_authenticated_full_backend_endpoint() {
        let request = prepare_shutdown_request("127.0.0.1:8765", "atlas-test-token");

        assert!(request.starts_with("POST /admin/prepare-shutdown HTTP/1.1\r\n"));
        assert!(request.contains("\r\nX-Atlas-Instance-Token: atlas-test-token\r\n"));
        assert!(request.ends_with("\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"));
    }
}
