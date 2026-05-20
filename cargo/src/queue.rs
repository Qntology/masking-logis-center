use tokio::sync::{mpsc, oneshot};
use crate::model::State;

// Define possible event types
pub enum Event {
    UpdateState {
        event_name: String,
        responder: oneshot::Sender<Result<(), anyhow::Error>>,
    },
    FetchState {
        responder: oneshot::Sender<Result<State, anyhow::Error>>,
    },
}

// Queue Orchestrator
pub struct TaskQueue {
    sender: mpsc::Sender<Event>,
}

impl TaskQueue {
    pub fn new(capacity: usize) -> (Self, TaskProcessor) {
        let (sender, receiver) = mpsc::channel(capacity);
        (TaskQueue { sender }, TaskProcessor { receiver })
    }

    pub async fn send_event(&self, event: Event) -> Result<(), anyhow::Error> {
        self.sender.send(event).await.map_err(|e| anyhow::anyhow!(e))
    }
}

pub struct TaskProcessor {
    receiver: mpsc::Receiver<Event>,
}

impl TaskProcessor {
    pub async fn run(mut self, state: State) {
        while let Some(event) = self.receiver.recv().await {
            match event {
                Event::UpdateState { event_name, responder } => {
                    // Logic to branch and process event (Phase 3 logic)
                    println!("Processing event: {}", event_name);
                    let _ = responder.send(Ok(()));
                }
                Event::FetchState { responder } => {
                    let _ = responder.send(Ok(state.clone()));
                }
            }
        }
    }
}
